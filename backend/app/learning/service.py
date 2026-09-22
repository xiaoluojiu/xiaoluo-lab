"""学习中心 · 服务层。

职责：把「目录（静态定义）」「判定（无状态计算）」「进度（持久化）」三层
组装成上层接口需要的形状，并保证状态迁移是幂等且可解释的。

状态机
------
    (无记录) --首次提交--> in_progress
    in_progress --检查全过--> completed
    completed  --继续改代码但没过--> 保持 completed（已完成是成就，不因后续
                                     实验性改动被剥夺；但仍记录新的 attempt）

这个取舍很重要：如果把 completed 回退成 in_progress，学习者就会因为一次
「想再试试别的写法」而失去完成标记，是典型的惩罚性设计。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundException, ValidationException
from app.learning.catalog import (
    CATALOG,
    TRACKS,
    ExperimentSpec,
    get_spec,
    spec_to_dict,
)
from app.learning.reviewer import review_submission
from app.models.learning import (
    CUSTOM_CARD_KINDS,
    LearningAttempt,
    LearningCustomCard,
    LearningProgress,
)

# 每个实验最多保留的过程记录条数（避免无限增长，也让「练习曲线」保持可读）
_MAX_ATTEMPTS = 30


def _normalize_checklist_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把前端传来的清单项归一化：过滤空文本、统一 shape、补 done 字段。

    前端可能传 {text} 或缺 done，这里做防御式规整，保证落库的 JSON 结构稳定。
    """
    out: list[dict[str, Any]] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        out.append({"text": text, "done": bool(raw.get("done", False))})
    return out


class LearningService:
    """学习中心业务服务。"""

    def __init__(self, db: Session | None = None) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # 目录
    # ------------------------------------------------------------------

    def list_tracks(self) -> list[dict[str, Any]]:
        return [dict(t) for t in TRACKS]

    def list_experiments(
        self,
        learner_id: str,
        *,
        track: str | None = None,
    ) -> list[dict[str, Any]]:
        """目录 + 该学习者进度（一次查询完成，避免 N+1）。"""
        progress_map = self._progress_map(learner_id)
        best_scores = self._best_scores_map(
            [r.id for r in progress_map.values() if r.status == "completed"]
        )
        items: list[dict[str, Any]] = []
        for spec in CATALOG:
            if track and spec.track != track:
                continue
            item = spec_to_dict(spec)
            row = progress_map.get(spec.key)
            item["progress"] = self._progress_summary(
                row, best_score=best_scores.get(row.id) if row else None
            )
            items.append(item)
        return items

    def get_experiment(self, learner_id: str, key: str) -> dict[str, Any]:
        spec = get_spec(key)
        if spec is None:
            raise NotFoundException(
                "learning experiment not found", details={"key": key}
            )
        row = self._get_progress(learner_id, key)
        data = spec_to_dict(spec)
        data["progress"] = self._progress_detail(row, spec)
        # 前置实验的完成情况，供前端决定是否提示「建议先做」
        if spec.prerequisite:
            pre = get_spec(spec.prerequisite)
            pre_row = self._get_progress(learner_id, spec.prerequisite)
            data["prerequisite_info"] = {
                "key": spec.prerequisite,
                "title": pre.title if pre else spec.prerequisite,
                "completed": bool(pre_row and pre_row.status == "completed"),
            }
        return data

    # ------------------------------------------------------------------
    # 检查 / 提交
    # ------------------------------------------------------------------

    def check(
        self,
        learner_id: str,
        key: str,
        code: str,
        df: pl.DataFrame | None = None,
        *,
        dataset_id: int | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """检查一次提交。

        ``persist=False`` 用于「只想看看哪里没过，先别记进进度」的场景，
        避免学习者每敲几行就污染练习曲线。
        """
        spec = get_spec(key)
        if spec is None:
            raise NotFoundException("learning experiment not found", details={"key": key})
        if not code.strip():
            raise ValidationException("提交的代码为空")

        result = review_submission(spec, code, df)
        payload = result.to_dict()

        if persist and self.db is not None:
            row = self._upsert_after_check(
                learner_id, spec, code, result, dataset_id=dataset_id
            )
            payload["progress"] = self._progress_detail(row, spec)
        else:
            payload["progress"] = self._progress_summary(
                self._get_progress(learner_id, key)
            )

        return payload

    def save_draft(
        self,
        learner_id: str,
        key: str,
        code: str,
        *,
        dataset_id: int | None = None,
    ) -> dict[str, Any]:
        """保存草稿：只落代码与数据集，不改完成状态、不增尝试次数。

        这是「保存草稿」这个按钮应该做的事——它此前是个纯前端假动作。
        """
        spec = get_spec(key)
        if spec is None:
            raise NotFoundException("learning experiment not found", details={"key": key})
        if self.db is None:
            raise ValidationException("草稿保存需要数据库会话")

        row = self._get_progress(learner_id, key)
        if row is None:
            row = LearningProgress(
                learner_id=learner_id,
                experiment_key=key,
                status="in_progress",
                checklist={},
                code=code,
                dataset_id=dataset_id,
                attempts=0,
            )
            self.db.add(row)
        else:
            row.code = code
            if dataset_id is not None:
                row.dataset_id = dataset_id
        self.db.commit()
        self.db.refresh(row)

        return {
            "saved": True,
            "saved_at": row.updated_at.isoformat() if row.updated_at else None,
            "progress": self._progress_summary(row),
        }

    def reset(self, learner_id: str, key: str) -> dict[str, Any]:
        """重置某个实验的进度（含过程记录）。"""
        spec = get_spec(key)
        if spec is None:
            raise NotFoundException("learning experiment not found", details={"key": key})
        row = self._get_progress(learner_id, key)
        if row is not None and self.db is not None:
            self.db.delete(row)
            self.db.commit()
        return {"reset": True, "key": key}

    # ------------------------------------------------------------------
    # 概览统计
    # ------------------------------------------------------------------

    def overview(self, learner_id: str) -> dict[str, Any]:
        """学习总览：完成度、各主线进度、最近活动。"""
        progress_map = self._progress_map(learner_id)
        best_scores = self._best_scores_map(
            [r.id for r in progress_map.values() if r.status == "completed"]
        )

        by_key: dict[str, dict[str, Any]] = {}
        track_stats: dict[str, dict[str, int]] = {
            t["id"]: {"total": 0, "completed": 0, "in_progress": 0} for t in TRACKS
        }
        level_stats: dict[str, int] = {}

        for spec in CATALOG:
            row = progress_map.get(spec.key)
            status = row.status if row else "not_started"
            by_key[spec.key] = {
                "status": status,
                "score": self._display_score(
                    row, best_score=best_scores.get(row.id) if row else None
                ),
                "attempts": row.attempts if row else 0,
            }
            stats = track_stats.setdefault(
                spec.track, {"total": 0, "completed": 0, "in_progress": 0}
            )
            stats["total"] += 1
            if status == "completed":
                stats["completed"] += 1
                level_stats[spec.level] = level_stats.get(spec.level, 0) + 1
            elif status == "in_progress":
                stats["in_progress"] += 1

        total = len(CATALOG)
        completed = sum(1 for v in by_key.values() if v["status"] == "completed")
        started = sum(1 for v in by_key.values() if v["status"] != "not_started")

        # 推荐下一个实验：未完成且前置已达成，按目录顺序取第一个
        recommended = None
        for spec in CATALOG:
            if by_key[spec.key]["status"] == "completed":
                continue
            if spec.prerequisite and by_key.get(spec.prerequisite, {}).get("status") != "completed":
                continue
            recommended = spec.key
            break

        return {
            "total": total,
            "started": started,
            "completed": completed,
            "completion_rate": int(round(completed / total * 100)) if total else 0,
            "tracks": [
                {"id": t["id"], "label": t["label"], **track_stats.get(t["id"], {})}
                for t in TRACKS
            ],
            "by_level": level_stats,
            "by_key": by_key,
            "recommended": recommended,
            "recent": self._recent_activity(learner_id, limit=6, best_scores=best_scores),
        }

    # ------------------------------------------------------------------
    # 自建学习卡片（与内置实验目录解耦，不做客观判分）
    # ------------------------------------------------------------------

    def list_custom_cards(self, learner_id: str) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        stmt = (
            select(LearningCustomCard)
            .where(LearningCustomCard.learner_id == learner_id)
            .order_by(LearningCustomCard.id.desc())
        )
        return [self._custom_card_to_dict(c) for c in self.db.scalars(stmt)]

    def get_custom_card(self, learner_id: str, card_id: int) -> dict[str, Any]:
        card = self._get_custom_card(learner_id, card_id)
        return self._custom_card_to_dict(card)

    def create_custom_card(
        self,
        learner_id: str,
        *,
        title: str,
        kind: str,
        goal: str = "",
        code: str = "",
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.db is None:
            raise ValidationException("创建卡片需要数据库会话")
        kind = kind.strip() or "practice"
        if kind not in CUSTOM_CARD_KINDS:
            raise ValidationException(
                f"卡片类型必须是 {' / '.join(CUSTOM_CARD_KINDS)} 之一",
                details={"kind": kind},
            )
        title = (title or "").strip()
        if not title:
            raise ValidationException("卡片标题不能为空")

        # 归一化清单项：过滤空文本，统一 shape，初始未勾选
        normalized_items = _normalize_checklist_items(items or [])

        card = LearningCustomCard(
            learner_id=learner_id,
            title=title[:128],
            kind=kind,
            goal=(goal or "").strip(),
            code=code or "",
            items=normalized_items,
        )
        self.db.add(card)
        self.db.commit()
        self.db.refresh(card)
        return self._custom_card_to_dict(card)

    def update_custom_card(
        self,
        learner_id: str,
        card_id: int,
        *,
        title: str | None = None,
        goal: str | None = None,
        code: str | None = None,
        items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """局部更新：只改传入的字段（None 表示不动）。"""
        card = self._get_custom_card(learner_id, card_id)
        if title is not None:
            t = title.strip()
            if not t:
                raise ValidationException("卡片标题不能为空")
            card.title = t[:128]
        if goal is not None:
            card.goal = goal.strip()
        if code is not None:
            card.code = code
        if items is not None:
            card.items = _normalize_checklist_items(items)
        assert self.db is not None
        self.db.commit()
        self.db.refresh(card)
        return self._custom_card_to_dict(card)

    def delete_custom_card(self, learner_id: str, card_id: int) -> dict[str, Any]:
        card = self._get_custom_card(learner_id, card_id)
        assert self.db is not None
        self.db.delete(card)
        self.db.commit()
        return {"deleted": True, "card_id": card_id}

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _get_custom_card(self, learner_id: str, card_id: int) -> LearningCustomCard:
        if self.db is None:
            raise ValidationException("读取卡片需要数据库会话")
        stmt = select(LearningCustomCard).where(
            LearningCustomCard.id == card_id,
            LearningCustomCard.learner_id == learner_id,
        )
        card = self.db.scalars(stmt).first()
        if card is None:
            raise NotFoundException(
                "learning custom card not found", details={"card_id": card_id}
            )
        return card

    @staticmethod
    def _custom_card_to_dict(card: LearningCustomCard) -> dict[str, Any]:
        items = list(card.items or [])
        done = sum(1 for it in items if it.get("done"))
        return {
            "id": card.id,
            "learner_id": card.learner_id,
            "title": card.title,
            "kind": card.kind,
            "goal": card.goal or "",
            "code": card.code or "",
            "items": items,
            "progress": {
                "done": done,
                "total": len(items),
                "rate": int(round(done / len(items) * 100)) if items else 0,
            },
            "created_at": card.created_at.isoformat() if card.created_at else None,
            "updated_at": card.updated_at.isoformat() if card.updated_at else None,
        }

    def _get_progress(self, learner_id: str, key: str) -> LearningProgress | None:
        if self.db is None:
            return None
        stmt = select(LearningProgress).where(
            LearningProgress.learner_id == learner_id,
            LearningProgress.experiment_key == key,
        )
        return self.db.scalars(stmt).first()

    def _progress_map(self, learner_id: str) -> dict[str, LearningProgress]:
        if self.db is None:
            return {}
        stmt = select(LearningProgress).where(LearningProgress.learner_id == learner_id)
        return {row.experiment_key: row for row in self.db.scalars(stmt)}

    def _upsert_after_check(
        self,
        learner_id: str,
        spec: ExperimentSpec,
        code: str,
        result: Any,
        *,
        dataset_id: int | None,
    ) -> LearningProgress:
        db = self.db
        assert db is not None

        row = self._get_progress(learner_id, spec.key)
        if row is None:
            row = LearningProgress(
                learner_id=learner_id,
                experiment_key=spec.key,
                status="in_progress",
                checklist={},
                code=code,
                dataset_id=dataset_id,
                attempts=0,
            )
            db.add(row)
            db.flush()

        # 逐条要求的达成情况，供前端把勾画在具体要求上
        achieved: dict[int, bool] = {i: True for i in range(len(spec.requirements))}
        for item in result.checks:
            if not item.passed:
                achieved[item.requirement_index] = False
        row.checklist = {
            spec.requirements[i]: achieved[i] for i in range(len(spec.requirements))
        }

        row.code = code
        if dataset_id is not None:
            row.dataset_id = dataset_id
        row.attempts = (row.attempts or 0) + 1
        row.last_passed = result.passed
        row.last_result = result.to_dict()

        if result.passed and row.status != "completed":
            row.status = "completed"
            row.completed_at = datetime.now()
        elif row.status != "completed":
            row.status = "in_progress"

        db.add(
            LearningAttempt(
                progress_id=row.id,
                score=result.score,
                passed=result.passed,
                checks=[c.to_dict() for c in result.checks],
                output_summary="; ".join(result.next_steps[:3]),
            )
        )
        db.flush()
        self._trim_attempts(row.id)
        db.commit()
        db.refresh(row)
        return row

    def _trim_attempts(self, progress_id: int) -> None:
        """只保留最近 N 次尝试，避免过程表无限膨胀。"""
        db = self.db
        assert db is not None
        stmt = (
            select(LearningAttempt)
            .where(LearningAttempt.progress_id == progress_id)
            .order_by(LearningAttempt.id.desc())
        )
        rows = list(db.scalars(stmt))
        for stale in rows[_MAX_ATTEMPTS:]:
            db.delete(stale)

    def _display_score(self, row: LearningProgress | None, *, best_score: int | None = None) -> int:
        """展示用的分数。

        取舍：**已完成的实验显示历史最高分，而不是最近一次的分**。
        学习者通关后再回去「试别的写法」是很自然的行为，如果这次没过就把
        卡片上的分数从那次的 100 改成 0，会给人「我退步了」的错觉，
        属于典型的惩罚性展示。未完成时才显示最近一次，因为它反映当前状态。

        ``best_score`` 由调用方批量预取（`_best_scores_map`），避免在
        ``overview`` / ``list_experiments`` 里逐实验查询造成 N+1。
        """
        if row is None:
            return 0
        last = (row.last_result or {}).get("score", 0)
        if row.status == "completed":
            best = best_score if best_score is not None else self._best_score(row.id)
            return max(last, best)
        return last

    def _best_score(self, progress_id: int) -> int:
        if self.db is None:
            return 0
        stmt = (
            select(LearningAttempt.score)
            .where(LearningAttempt.progress_id == progress_id)
            .order_by(LearningAttempt.score.desc())
            .limit(1)
        )
        return self.db.scalars(stmt).first() or 0

    def _best_scores_map(self, progress_ids: list[int]) -> dict[int, int]:
        """一次查询出所有 progress 的最高分，供批量评分用（消除 N+1）。"""
        if self.db is None or not progress_ids:
            return {}
        stmt = (
            select(LearningAttempt.progress_id, func.max(LearningAttempt.score))
            .where(LearningAttempt.progress_id.in_(progress_ids))
            .group_by(LearningAttempt.progress_id)
        )
        return {pid: int(score or 0) for pid, score in self.db.execute(stmt)}

    def _progress_summary(
        self, row: LearningProgress | None, *, best_score: int | None = None
    ) -> dict[str, Any]:
        if row is None:
            return {
                "status": "not_started",
                "attempts": 0,
                "score": 0,
                "last_passed": None,
                "completed_at": None,
                "saved_at": None,
            }
        return {
            "status": row.status,
            "attempts": row.attempts or 0,
            "score": self._display_score(row, best_score=best_score),
            "last_passed": row.last_passed,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "saved_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    def _progress_detail(
        self, row: LearningProgress | None, spec: ExperimentSpec
    ) -> dict[str, Any]:
        base = self._progress_summary(row)
        base["code"] = row.code if row else spec.starter
        base["dataset_id"] = row.dataset_id if row else None
        base["checklist"] = dict(row.checklist or {}) if row else {}
        base["last_result"] = dict(row.last_result or {}) if row else {}
        base["history"] = self._attempt_history(row)
        return base

    def _attempt_history(self, row: LearningProgress | None) -> list[dict[str, Any]]:
        if row is None or self.db is None:
            return []
        stmt = (
            select(LearningAttempt)
            .where(LearningAttempt.progress_id == row.id)
            .order_by(LearningAttempt.id.desc())
            .limit(_MAX_ATTEMPTS)
        )
        return [
            {
                "score": a.score,
                "passed": a.passed,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "summary": a.output_summary,
            }
            for a in self.db.scalars(stmt)
        ]

    def _recent_activity(
        self,
        learner_id: str,
        limit: int = 6,
        *,
        best_scores: dict[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        stmt = (
            select(LearningProgress)
            .where(LearningProgress.learner_id == learner_id)
            .order_by(LearningProgress.updated_at.desc())
            .limit(limit)
        )
        out: list[dict[str, Any]] = []
        for row in self.db.scalars(stmt):
            spec = get_spec(row.experiment_key)
            out.append(
                {
                    "key": row.experiment_key,
                    "title": spec.title if spec else row.experiment_key,
                    "track": spec.track if spec else "ml",
                    "status": row.status,
                    "score": self._display_score(
                        row, best_score=(best_scores or {}).get(row.id)
                    ),
                    "attempts": row.attempts or 0,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                }
            )
        return out
