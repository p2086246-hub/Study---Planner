"""
ML Engine — Smart Study Planner (Enhanced)
- Linear Regression for study hour prediction
- Priority scoring
- 7-day timetable: break slots, no consecutive same subject, no mock test
- Subject color palette
- Badge checker
- Weak subject detector
- AI suggestion generator
- Smart time validator
"""

import numpy as np
from datetime import date, timedelta
import random

# ── Subject colors ────────────────────────────────────────────────────────
SUBJECT_COLORS = [
    "#4f46e5","#0891b2","#059669","#d97706",
    "#dc2626","#7c3aed","#db2777","#65a30d",
    "#ea580c","#0284c7",
]

def get_subject_color(index):
    return SUBJECT_COLORS[index % len(SUBJECT_COLORS)]


# ── 1. Linear Regression ──────────────────────────────────────────────────
class StudyHourPredictor:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0
        self._train()

    def _make_data(self, n=600):
        rng = np.random.default_rng(42)
        difficulty    = rng.uniform(1, 10, n)
        days_left     = rng.uniform(3, 90, n)
        concentration = rng.uniform(1, 10, n)
        hours = (0.35*difficulty + 8.0/days_left - 0.12*concentration + 0.8
                 + rng.normal(0, 0.2, n))
        hours = np.clip(hours, 0.5, 6.0)
        return np.column_stack([difficulty, days_left, concentration]), hours

    def _train(self):
        X, y = self._make_data()
        Xa   = np.hstack([np.ones((X.shape[0],1)), X])
        beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        self.intercept_ = beta[0]
        self.coef_      = beta[1:]

    def predict(self, difficulty, days_left, concentration):
        days_left = max(1, days_left)
        h = self.intercept_ + np.dot(self.coef_,
              np.array([difficulty, days_left, concentration]))
        return round(float(np.clip(h, 0.5, 6.0)), 2)

    def r2_score(self):
        X, y = self._make_data(200)
        preds = np.array([self.predict(r[0],r[1],r[2]) for r in X])
        ss_res = np.sum((y-preds)**2)
        ss_tot = np.sum((y-y.mean())**2)
        return round(1 - ss_res/ss_tot, 4)


# ── 2. Priority ───────────────────────────────────────────────────────────
def compute_priority(difficulty, days_left):
    days_left = max(1, days_left)
    return round(difficulty*0.6 + (1.0/days_left)*0.4, 4)


# ── 3. Smart Time Validator ───────────────────────────────────────────────
def validate_schedule_hours(start_time_str, free_hours):
    """
    Ensures the schedule is realistic:
    - Max free_hours capped at 8
    - No schedule ending after 23:00
    - Start time should be between 05:00 and 21:00
    Returns (validated_hours, warning_message_or_None)
    """
    try:
        h, m = map(int, start_time_str.split(':'))
    except Exception:
        h, m = 18, 0

    # Cap hours
    free_hours = min(float(free_hours), 8.0)

    # Check end time
    end_hour = h + int(free_hours)
    warning  = None

    if h < 5:
        warning = "Study start time before 5 AM is not recommended. Adjusted to 06:00."
        h = 6
    if h > 21:
        warning = "Study start time after 9 PM is too late. Adjusted to 18:00."
        h = 18
    if end_hour > 23:
        max_hours = 23 - h
        free_hours = max(1, max_hours)
        warning = (f"Schedule would end after midnight. "
                   f"Hours reduced to {free_hours}h to keep it realistic.")

    start_time_str = f"{h:02d}:{m:02d}"
    return free_hours, start_time_str, warning


# ── 4. No-consecutive shuffle ─────────────────────────────────────────────
def _no_consecutive(queue):
    shuffled = queue[:]
    random.seed(42)
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        if shuffled[i] == shuffled[i-1]:
            for j in range(i+1, len(shuffled)):
                if shuffled[j] != shuffled[i-1]:
                    shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
                    break
    return shuffled


# ── 5. Timetable Generator ────────────────────────────────────────────────
DAYS = ['MON','TUE','WED','THU','FRI','SAT','SUN']

def _parse_time(t_str):
    h, m = t_str.split(':')
    return int(h), int(m)

def _fmt_slot(h, m):
    suffix = 'AM' if h < 12 else 'PM'
    hh = h if h <= 12 else h - 12
    if hh == 0: hh = 12
    return f"{hh:02d}:{m:02d} {suffix}"


def generate_timetable(subjects, profile, free_hours_override=None, use_topics=False):
    """
    subjects: list of Subject ORM objects (or Topic-like dicts with .name, .priority_score)
    use_topics: if True, subjects list contains topic-level items
    Returns list of dicts for WeeklyTimetable.
    Slot types: Study, Revision, Break. NO mock test.
    Break after every 2 study slots. No consecutive same subject.
    """
    if not subjects or not profile:
        return []

    base_hours = float(free_hours_override or profile.daily_free_hours or 0)

    # Smart boost for close exams (only in subject mode)
    if not use_topics:
        min_days = min((s.days_left for s in subjects), default=30)
        if min_days <= 3:   boost = 2
        elif min_days <= 7: boost = 1
        else:               boost = 0
        base_hours = min(base_hours + boost, 8.0)

    # Validate realistic schedule
    base_hours, start_str, _ = validate_schedule_hours(
        profile.study_start_time or '18:00', base_hours)

    study_hours = int(base_hours)
    if study_hours < 1:
        return []

    start_h, start_m = _parse_time(start_str)

    # Pattern: Study, Study, Break, Study, Study, Break … Revision
    break_slots    = study_hours // 2
    total_physical = study_hours + break_slots + 1  # +1 for revision

    # Build time labels
    time_labels = []
    ch = start_h
    for _ in range(total_physical):
        eh = (ch + 1) % 24
        time_labels.append(f"{_fmt_slot(ch, start_m)} – {_fmt_slot(eh, start_m)}")
        ch = eh

    # Build pattern
    pattern, sc, sp, bp = [], 0, 0, 0
    while sp < study_hours:
        pattern.append('Study'); sp += 1; sc += 1
        if sc == 2 and bp < break_slots and sp < study_hours:
            pattern.append('Break'); bp += 1; sc = 0
    pattern.append('Revision')
    time_labels = time_labels[:len(pattern)]

    # Weighted subject queue
    total_study = study_hours * 7
    priorities  = np.array([max(getattr(s,'priority_score',0.01) or 0.01, 0.01) for s in subjects])
    weights     = priorities / priorities.sum()
    raw_slots   = np.round(weights * total_study).astype(int)
    raw_slots   = np.maximum(raw_slots, 1)
    diff = total_study - raw_slots.sum()
    if diff > 0: raw_slots[np.argmax(weights)] += diff
    elif diff < 0:
        for _ in range(-diff): raw_slots[np.argmax(raw_slots-1)] -= 1

    subj_queue = []
    for subj, cnt in zip(subjects, raw_slots):
        subj_queue.extend([subj.name] * int(cnt))
    subj_queue = _no_consecutive(subj_queue)

    today      = date.today()
    week_start = today - timedelta(days=today.weekday())
    records, q = [], 0

    for d_idx, day in enumerate(DAYS):
        for s_idx, (tlabel, stype) in enumerate(zip(time_labels, pattern)):
            if stype == 'Revision':
                records.append(dict(week_start=week_start, day_name=day,
                    time_slot=tlabel, subject_name='Revision', slot_type='Revision'))
            elif stype == 'Break':
                records.append(dict(week_start=week_start, day_name=day,
                    time_slot=tlabel, subject_name='Break', slot_type='Break'))
            else:
                sname = subj_queue[q % len(subj_queue)]; q += 1
                records.append(dict(week_start=week_start, day_name=day,
                    time_slot=tlabel, subject_name=sname, slot_type='Study'))
    return records


# ── 6. Backlog reschedule ─────────────────────────────────────────────────
def get_backlog_slots(user_id, db, WeeklyTimetable, DAYS_list):
    """
    Returns list of missed (uncompleted Study) slots from past days this week.
    """
    from datetime import date, timedelta
    today      = date.today()
    week_start = today - timedelta(days=today.weekday())
    today_name = DAYS_list[today.weekday()]

    past_days  = DAYS_list[:today.weekday()]  # days before today
    missed     = []
    for day in past_days:
        slots = WeeklyTimetable.query.filter_by(
            user_id=user_id, week_start=week_start,
            day_name=day, slot_type='Study', completed=False).all()
        missed.extend(slots)
    return missed


# ── 7. Badge checker ──────────────────────────────────────────────────────
def check_badges(streak, done_slots, total_slots, subjects):
    badges = []
    if streak >= 7:
        badges.append({"name":"7-Day Streak","icon":"🔥","desc":"Studied 7 days in a row!","color":"orange"})
    if streak >= 3:
        badges.append({"name":"3-Day Streak","icon":"⚡","desc":"3 days straight!","color":"yellow"})
    if total_slots > 0 and done_slots == total_slots:
        badges.append({"name":"All Done!","icon":"🏆","desc":"Completed every slot this week!","color":"green"})
    if total_slots > 0 and done_slots >= total_slots * 0.5:
        badges.append({"name":"Halfway There","icon":"💪","desc":"50% of this week done.","color":"blue"})
    if len(subjects) >= 5:
        badges.append({"name":"Subject Master","icon":"📚","desc":"Managing 5+ subjects!","color":"purple"})
    if any(s.days_left <= 7 for s in subjects):
        badges.append({"name":"Exam Mode","icon":"🎯","desc":"Exam within 7 days!","color":"red"})
    return badges


# ── 8. Weak subject detector ──────────────────────────────────────────────
def detect_weak_subjects(subjects, week_slots):
    """
    Returns list of {name, reason, color} for subjects needing attention.
    Criteria: high difficulty (>=7) OR low completion rate (<40%) OR days_left<14
    """
    weak = []
    done_map = {}
    total_map = {}
    for sl in week_slots:
        if sl.slot_type == 'Study':
            total_map[sl.subject_name] = total_map.get(sl.subject_name, 0) + 1
            if sl.completed:
                done_map[sl.subject_name] = done_map.get(sl.subject_name, 0) + 1

    for s in subjects:
        reasons = []
        total = total_map.get(s.name, 0)
        done  = done_map.get(s.name, 0)
        rate  = (done / total * 100) if total > 0 else 0

        if s.difficulty >= 7:
            reasons.append(f"High difficulty ({s.difficulty}/10)")
        if total > 0 and rate < 40:
            reasons.append(f"Low completion ({int(rate)}%)")
        if s.days_left and s.days_left < 14:
            reasons.append(f"Exam in {s.days_left} days")

        if reasons:
            weak.append({"name": s.name, "reason": ", ".join(reasons),
                         "color": s.color or "#dc2626",
                         "days_left": s.days_left or 999})
    weak.sort(key=lambda x: x["days_left"])
    return weak


# ── 9. AI Suggestion generator ────────────────────────────────────────────
def generate_ai_suggestions(subjects, week_slots, streak, missed_backlog):
    suggestions = []

    # Exam urgency
    for s in subjects:
        if hasattr(s, 'days_left') and s.days_left and s.days_left <= 7:
            suggestions.append({
                "icon": "🚨",
                "text": f"Your exam for <strong>{s.name}</strong> is in {s.days_left} days — increase study time!",
                "type": "danger"
            })

    # Missed sessions
    missed_counts = {}
    for sl in missed_backlog:
        missed_counts[sl.subject_name] = missed_counts.get(sl.subject_name, 0) + 1
    for sname, cnt in missed_counts.items():
        suggestions.append({
            "icon": "⚠️",
            "text": f"You missed <strong>{cnt}</strong> session(s) of <strong>{sname}</strong> — rescheduled to upcoming days.",
            "type": "warning"
        })

    # Streak
    if streak == 0:
        suggestions.append({"icon": "💡",
            "text": "Start your streak today — complete all slots to begin tracking!",
            "type": "info"})
    elif streak >= 5:
        suggestions.append({"icon": "🔥",
            "text": f"Amazing! You're on a <strong>{streak}-day streak</strong>. Keep it up!",
            "type": "success"})

    # Revision topics (subjects with days_left < 14)
    for s in subjects:
        if hasattr(s, 'days_left') and s.days_left and 7 < s.days_left <= 14:
            suggestions.append({"icon": "📝",
                "text": f"Consider revising <strong>{s.name}</strong> tomorrow — exam in {s.days_left} days.",
                "type": "info"})

    # Cap at 5 suggestions
    return suggestions[:5]


# Singleton
predictor = StudyHourPredictor()
