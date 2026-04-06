"""
Smart Study Planner — Enhanced Flask App
New: topic-wise planning, backlog tracker, daily goals, weak subject detection,
     AI suggestions, browser notifications, smart time control.
"""

import json
from datetime import datetime, date, timedelta

from flask import (Flask, render_template, redirect, url_for,
                   request, flash, jsonify)
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

from models import (db, User, Profile, Subject, Topic, WeeklyTimetable,
                    Streak, DailyGoal, ContactMessage)
from ml_engine import (predictor, compute_priority, generate_timetable,
                       DAYS, check_badges, get_subject_color, detect_weak_subjects,
                       generate_ai_suggestions, get_backlog_slots,
                       validate_schedule_hours)

# ── Motivations ───────────────────────────────────────────────────────────
MOTIVATIONS = [
    "The secret of getting ahead is getting started. — Mark Twain",
    "It always seems impossible until it's done. — Nelson Mandela",
    "Don't watch the clock; do what it does. Keep going.",
    "Success is the sum of small efforts repeated day in and day out.",
    "Your future is created by what you do today, not tomorrow.",
    "Hard work beats talent when talent doesn't work hard.",
    "Believe you can and you're halfway there. — Theodore Roosevelt",
    "Study now, shine later. Every hour counts.",
    "Push yourself, because no one else is going to do it for you.",
    "Dream big. Work hard. Stay focused.",
    "The expert in anything was once a beginner.",
    "Education is the most powerful weapon you can use to change the world.",
    "Small daily improvements over time lead to stunning results.",
    "You don't have to be great to start, but you have to start to be great.",
    "Strive for progress, not perfection.",
]

def get_motivation():
    return MOTIVATIONS[date.today().timetuple().tm_yday % len(MOTIVATIONS)]

# ── App ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]                  = "ssp_secret_2024_xK9"
app.config["SQLALCHEMY_DATABASE_URI"]     = "sqlite:///study_planner.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

from datetime import timedelta as _td
@app.context_processor
def inject_globals():
    return dict(timedelta=_td)

login_manager = LoginManager(app)
login_manager.login_view             = "login"
login_manager.login_message          = "Please log in to access this page."
login_manager.login_message_category = "info"

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# ── Helpers ───────────────────────────────────────────────────────────────

def get_or_create_profile(user_id):
    p = Profile.query.filter_by(user_id=user_id).first()
    if not p:
        p = Profile(user_id=user_id)
        db.session.add(p); db.session.commit()
    return p

def refresh_subject_scores(user_id):
    prof     = get_or_create_profile(user_id)
    subjects = Subject.query.filter_by(user_id=user_id).all()
    conc     = prof.concentration_level or 7
    for i, s in enumerate(subjects):
        if s.exam_date:
            s.days_left = max(1, (s.exam_date - date.today()).days)
        s.priority_score  = compute_priority(s.difficulty, s.days_left)
        s.predicted_hours = predictor.predict(s.difficulty, s.days_left, conc)
        if not s.color:
            s.color = get_subject_color(i)
    db.session.commit()
    return subjects

def get_current_streak(user_id):
    latest = (Streak.query.filter_by(user_id=user_id)
              .order_by(Streak.date.desc()).first())
    return latest.current_streak if latest else 0

def build_timetable_grid(slots):
    slot_order, seen = [], set()
    for s in slots:
        if s.time_slot not in seen:
            slot_order.append(s.time_slot); seen.add(s.time_slot)
    grid = {ts: {d: None for d in DAYS} for ts in slot_order}
    for s in slots:
        grid[s.time_slot][s.day_name] = s
    return slot_order, grid

def smart_hours(daily_free_hours, subjects):
    urgent = [s for s in subjects if s.days_left < 14]
    if urgent and daily_free_hours < 4:
        return daily_free_hours + min(2.0, 4.0 - daily_free_hours)
    return daily_free_hours

def get_subject_color_map(subjects):
    cmap = {s.name: (s.color or get_subject_color(i)) for i, s in enumerate(subjects)}
    cmap['Revision'] = '#92400e'
    cmap['Break']    = '#374151'
    return cmap

def _process_backlog(user_id):
    """Move uncompleted past-day Study slots to today as backlog."""
    today      = date.today()
    week_start = today - timedelta(days=today.weekday())
    today_name = DAYS[today.weekday()]
    if today.weekday() == 0:   # Monday — no past days this week
        return 0
    past_days  = DAYS[:today.weekday()]
    missed = WeeklyTimetable.query.filter(
        WeeklyTimetable.user_id   == user_id,
        WeeklyTimetable.week_start == week_start,
        WeeklyTimetable.day_name.in_(past_days),
        WeeklyTimetable.slot_type  == 'Study',
        WeeklyTimetable.completed  == False,
        WeeklyTimetable.is_backlog == False
    ).all()
    # Get today's existing slots count
    today_count = WeeklyTimetable.query.filter_by(
        user_id=user_id, week_start=week_start, day_name=today_name).count()
    added = 0
    for sl in missed[:3]:   # max 3 backlog slots per day to avoid overload
        if today_count + added >= 8:
            break
        # Mark original as backlog-processed so we don't duplicate
        sl.is_backlog = True
        # Find a suitable time slot (append after today's last slot)
        today_slots = WeeklyTimetable.query.filter_by(
            user_id=user_id, week_start=week_start, day_name=today_name
        ).order_by(WeeklyTimetable.time_slot.desc()).first()
        new_slot = WeeklyTimetable(
            user_id=user_id, week_start=week_start,
            day_name=today_name, time_slot="Backlog",
            subject_name=sl.subject_name,
            slot_type='Study', is_backlog=True
        )
        db.session.add(new_slot)
        added += 1
    db.session.commit()
    return added

# ── Public routes ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("admin_panel") if current_user.role == "admin"
                        else url_for("dashboard"))
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET","POST"])
def contact():
    if request.method == "POST":
        name  = request.form.get("name","").strip()
        email = request.form.get("email","").strip()
        subj  = request.form.get("subject","").strip()
        msg   = request.form.get("message","").strip()
        if name and email and msg:
            db.session.add(ContactMessage(name=name, email=email,
                                          subject=subj, message=msg))
            db.session.commit()
            flash("Message sent!", "success")
        else:
            flash("Please fill all required fields.", "danger")
        return redirect(url_for("contact"))
    return render_template("contact.html")

# ── Auth ──────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET","POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form["username"].strip()
        email    = request.form["email"].strip().lower()
        password = request.form["password"]
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))
        user = User(username=username, email=email,
                    password_hash=generate_password_hash(password))
        db.session.add(user); db.session.flush()
        db.session.add(Profile(user_id=user.id, full_name=username))
        db.session.commit()
        flash("Account created! Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("admin_panel") if current_user.role == "admin"
                        else url_for("dashboard"))
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f"Welcome back, {user.username}!", "success")
            nxt = request.args.get("next")
            if nxt: return redirect(nxt)
            return redirect(url_for("admin_panel") if user.role == "admin"
                            else url_for("dashboard"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))

# ── Dashboard ─────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    profile  = get_or_create_profile(current_user.id)
    subjects = refresh_subject_scores(current_user.id)
    streak   = get_current_streak(current_user.id)

    today      = date.today()
    week_start = today - timedelta(days=today.weekday())
    day_name   = DAYS[today.weekday()]

    # Auto-process backlog
    backlog_added = _process_backlog(current_user.id)
    if backlog_added:
        flash(f"📋 {backlog_added} missed slot(s) rescheduled to today as backlog.", "warning")

    today_slots = (WeeklyTimetable.query
                   .filter_by(user_id=current_user.id,
                               week_start=week_start, day_name=day_name)
                   .order_by(WeeklyTimetable.time_slot).all())

    week_slots = WeeklyTimetable.query.filter_by(
        user_id=current_user.id, week_start=week_start).all()
    total_w  = len(week_slots)
    done_w   = sum(1 for s in week_slots if s.completed)
    week_pct = round(done_w/total_w*100, 1) if total_w else 0

    # Daily goals
    today_goals = DailyGoal.query.filter_by(
        user_id=current_user.id, date=today).order_by(DailyGoal.id).all()

    # Badges
    badges = check_badges(streak, done_w, total_w, subjects)

    # Exam countdowns
    countdowns = []
    for s in subjects:
        if s.exam_date:
            diff = (s.exam_date - today).days
            countdowns.append({"name": s.name, "days_left": max(0,diff),
                                "color": s.color or "#4f46e5", "urgent": diff<=7})
    countdowns.sort(key=lambda x: x["days_left"])

    # Weak subjects
    weak_subjects = detect_weak_subjects(subjects, week_slots)

    # Missed backlog list for suggestions
    missed_backlog = get_backlog_slots(current_user.id, db, WeeklyTimetable, DAYS)

    # AI suggestions
    ai_suggestions = generate_ai_suggestions(subjects, week_slots, streak, missed_backlog)

    color_map = get_subject_color_map(subjects)

    # Upcoming exam notifications data
    exam_alerts = [{"name": s.name, "days": (s.exam_date - today).days}
                   for s in subjects if s.exam_date and 0 < (s.exam_date - today).days <= 7]

    return render_template("dashboard.html",
        profile=profile, subjects=subjects, streak=streak,
        today_slots=today_slots, day_name=day_name,
        subj_names=json.dumps([s.name for s in subjects]),
        subj_prio=json.dumps([round(s.priority_score,3) for s in subjects]),
        subj_colors=json.dumps([s.color or "#4f46e5" for s in subjects]),
        week_pct=week_pct, done_w=done_w, total_w=total_w,
        model_r2=predictor.r2_score(),
        motivation=get_motivation(),
        today_goals=today_goals,
        badges=badges,
        countdowns=countdowns,
        weak_subjects=weak_subjects,
        ai_suggestions=ai_suggestions,
        color_map=color_map,
        exam_alerts=json.dumps(exam_alerts))


@app.route("/mark_slot/<int:slot_id>", methods=["POST"])
@login_required
def mark_slot(slot_id):
    slot = WeeklyTimetable.query.get_or_404(slot_id)
    if slot.user_id != current_user.id:
        return jsonify({"error":"forbidden"}), 403
    slot.completed = not slot.completed
    db.session.commit()
    _update_streak()
    return jsonify({"completed": slot.completed})

def _update_streak():
    today      = date.today()
    week_start = today - timedelta(days=today.weekday())
    day_name   = DAYS[today.weekday()]
    slots = WeeklyTimetable.query.filter_by(
        user_id=current_user.id, week_start=week_start, day_name=day_name).all()
    # Only count Study and Revision for streak (not breaks)
    countable = [s for s in slots if s.slot_type in ('Study','Revision')]
    all_done  = bool(countable) and all(s.completed for s in countable)
    rec  = Streak.query.filter_by(user_id=current_user.id, date=today).first()
    prev = get_current_streak(current_user.id)
    if not rec:
        rec = Streak(user_id=current_user.id, date=today)
        db.session.add(rec)
    rec.all_completed  = all_done
    rec.current_streak = (prev+1) if all_done else 0
    db.session.commit()

# ── Daily Goals ───────────────────────────────────────────────────────────

@app.route("/goal/add", methods=["POST"])
@login_required
def add_goal():
    text = request.form.get("goal_text","").strip()
    if text:
        today_goals = DailyGoal.query.filter_by(
            user_id=current_user.id, date=date.today()).count()
        if today_goals >= 5:
            flash("Maximum 5 daily goals allowed.", "warning")
        else:
            db.session.add(DailyGoal(user_id=current_user.id,
                                     date=date.today(), text=text))
            db.session.commit()
    return redirect(url_for("dashboard"))

@app.route("/goal/toggle/<int:goal_id>", methods=["POST"])
@login_required
def toggle_goal(goal_id):
    g = DailyGoal.query.get_or_404(goal_id)
    if g.user_id != current_user.id:
        return jsonify({"error":"forbidden"}), 403
    g.completed = not g.completed
    db.session.commit()
    return jsonify({"completed": g.completed})

@app.route("/goal/delete/<int:goal_id>", methods=["POST"])
@login_required
def delete_goal(goal_id):
    g = DailyGoal.query.get_or_404(goal_id)
    if g.user_id == current_user.id:
        db.session.delete(g); db.session.commit()
    return redirect(url_for("dashboard"))

# ── Profile ───────────────────────────────────────────────────────────────

@app.route("/profile", methods=["GET","POST"])
@login_required
def profile():
    prof = get_or_create_profile(current_user.id)
    if request.method == "POST":
        prof.full_name           = request.form["full_name"].strip()
        prof.age                 = int(request.form.get("age", 20))
        prof.education_level     = request.form.get("education_level","").strip()
        prof.study_goal          = request.form.get("study_goal","").strip()
        prof.concentration_level = int(request.form.get("concentration_level", 7))
        db.session.commit()
        flash("Profile saved!", "success")
        return redirect(url_for("profile"))
    return render_template("profile.html", profile=prof, user=current_user)

# ── Study Planner ─────────────────────────────────────────────────────────

@app.route("/planner", methods=["GET","POST"])
@login_required
def planner():
    profile  = get_or_create_profile(current_user.id)
    subjects = Subject.query.filter_by(user_id=current_user.id).all()
    mode     = request.args.get("mode", profile.planner_mode or "subject")

    if request.method == "POST":
        action = request.form.get("action")

        # ── Set planner mode ──────────────────────────────────────────────
        if action == "set_mode":
            new_mode = request.form.get("mode","subject")
            profile.planner_mode = new_mode
            db.session.commit()
            return redirect(url_for("planner", mode=new_mode))

        # ── Add subject ───────────────────────────────────────────────────
        elif action == "add_subject":
            name       = request.form["name"].strip()
            difficulty = float(request.form.get("difficulty", 5))
            exam_str   = request.form.get("exam_date","").strip()
            exam_date  = None; days_left = 30
            if exam_str:
                try:
                    exam_date = datetime.strptime(exam_str, "%Y-%m-%d").date()
                    days_left = max(1, (exam_date - date.today()).days)
                except ValueError:
                    pass
            conc     = profile.concentration_level or 7
            color    = get_subject_color(Subject.query.filter_by(user_id=current_user.id).count())
            subj     = Subject(user_id=current_user.id, name=name,
                               difficulty=difficulty, exam_date=exam_date,
                               days_left=days_left, color=color,
                               priority_score=compute_priority(difficulty, days_left),
                               predicted_hours=predictor.predict(difficulty, days_left, conc))
            db.session.add(subj); db.session.commit()
            flash(f"Subject '{name}' added!", "success")

        # ── Delete subject ────────────────────────────────────────────────
        elif action == "delete_subject":
            sid = int(request.form["subject_id"])
            s   = Subject.query.get(sid)
            if s and s.user_id == current_user.id:
                db.session.delete(s); db.session.commit()
                flash("Subject deleted.", "info")

        # ── Add topic ─────────────────────────────────────────────────────
        elif action == "add_topic":
            subject_id = int(request.form.get("subject_id", 0))
            tname      = request.form.get("topic_name","").strip()
            tdiff      = float(request.form.get("topic_difficulty", 5))
            if tname and subject_id:
                t = Topic(subject_id=subject_id, user_id=current_user.id,
                          name=tname, difficulty=tdiff, status="pending")
                db.session.add(t); db.session.commit()
                flash(f"Topic '{tname}' added!", "success")

        # ── Update topic status ───────────────────────────────────────────
        elif action == "update_topic_status":
            tid    = int(request.form.get("topic_id", 0))
            status = request.form.get("status","pending")
            t      = Topic.query.get(tid)
            if t and t.user_id == current_user.id:
                t.status = status; db.session.commit()
                flash("Topic status updated.", "success")

        # ── Delete topic ──────────────────────────────────────────────────
        elif action == "delete_topic":
            tid = int(request.form.get("topic_id", 0))
            t   = Topic.query.get(tid)
            if t and t.user_id == current_user.id:
                db.session.delete(t); db.session.commit()
                flash("Topic deleted.", "info")

        # ── Generate timetable ────────────────────────────────────────────
        elif action == "generate":
            free_hours = float(request.form.get("free_hours", 0))
            start_time = request.form.get("start_time", "18:00").strip()
            end_time   = request.form.get("end_time",   "22:00").strip()

            if free_hours < 1:
                flash("Please enter daily free hours (minimum 1).", "danger")
                return redirect(url_for("planner", mode=mode))

            # Smart time validation
            free_hours, start_time, time_warn = validate_schedule_hours(start_time, free_hours)
            if time_warn:
                flash(f"⏰ {time_warn}", "warning")

            subjects_ref = refresh_subject_scores(current_user.id)
            use_topics   = (mode == "topic")

            if use_topics:
                # Build topic-like objects with priority_score
                all_topics = Topic.query.filter_by(
                    user_id=current_user.id, status="pending").all()
                if len(all_topics) < 3:
                    flash("Add at least 3 pending topics before generating in topic mode.", "warning")
                    return redirect(url_for("planner", mode=mode))
                # Give topics a priority_score based on parent subject
                subj_map = {s.id: s for s in subjects_ref}
                for t in all_topics:
                    parent = subj_map.get(t.subject_id)
                    days   = parent.days_left if parent else 30
                    t.priority_score = compute_priority(t.difficulty, days)
                gen_items = all_topics
            else:
                if len(subjects_ref) < 3:
                    flash("Add at least 3 subjects before generating.", "warning")
                    return redirect(url_for("planner", mode=mode))
                effective = smart_hours(free_hours, subjects_ref)
                if effective > free_hours:
                    flash(f"⚡ Exams are close — hours boosted to {effective}h/day.", "warning")
                    free_hours = effective
                gen_items = subjects_ref

            profile.daily_free_hours = free_hours
            profile.study_start_time = start_time
            profile.study_end_time   = end_time
            db.session.commit()

            today      = date.today()
            week_start = today - timedelta(days=today.weekday())
            WeeklyTimetable.query.filter_by(
                user_id=current_user.id, week_start=week_start).delete()
            db.session.commit()

            records = generate_timetable(gen_items, profile,
                                         free_hours_override=free_hours,
                                         use_topics=use_topics)
            for r in records:
                r["user_id"] = current_user.id
                db.session.add(WeeklyTimetable(**r))
            db.session.commit()

            study_c = sum(1 for r in records if r["slot_type"]=="Study")
            break_c = sum(1 for r in records if r["slot_type"]=="Break")
            flash(f"✅ Timetable generated! {study_c} study + {break_c} break slots.", "success")

        return redirect(url_for("planner", mode=mode))

    # GET
    subjects_list = refresh_subject_scores(current_user.id)
    topics_by_subj = {}
    for s in subjects_list:
        topics_by_subj[s.id] = Topic.query.filter_by(
            subject_id=s.id, user_id=current_user.id).all()

    today      = date.today()
    week_start = today - timedelta(days=today.weekday())
    slots      = WeeklyTimetable.query.filter_by(
        user_id=current_user.id, week_start=week_start).all()
    slot_order, grid = build_timetable_grid(slots)
    color_map  = get_subject_color_map(subjects_list)

    return render_template("planner.html",
        profile=profile, subjects=subjects_list,
        topics_by_subj=topics_by_subj,
        days=DAYS, slot_order=slot_order, grid=grid,
        week_start=week_start,
        free_hours=profile.daily_free_hours or '',
        color_map=color_map,
        mode=mode)

# ── Streak Tracker ────────────────────────────────────────────────────────

@app.route("/streak")
@login_required
def streak_tracker():
    streaks = (Streak.query.filter_by(user_id=current_user.id)
               .order_by(Streak.date.desc()).limit(30).all())
    current = get_current_streak(current_user.id)
    best    = max((s.current_streak for s in streaks), default=0)
    today   = date.today()
    heatmap = []
    for i in range(13,-1,-1):
        d   = today - timedelta(days=i)
        rec = next((s for s in streaks if s.date==d), None)
        heatmap.append({"date": d.strftime("%b %d"),
                        "done": bool(rec and rec.all_completed)})
    return render_template("streak.html",
        streaks=streaks, current_streak=current,
        best_streak=best, heatmap=heatmap)

# ── Admin ─────────────────────────────────────────────────────────────────

@app.route("/admin")
@login_required
def admin_panel():
    if current_user.role != "admin":
        flash("Access denied.", "danger")
        return redirect(url_for("dashboard"))
    users    = User.query.all()
    messages = ContactMessage.query.order_by(ContactMessage.created_at.desc()).all()
    return render_template("admin/panel.html",
        users=users, messages=messages,
        total_subjects=Subject.query.count(),
        total_timetable=WeeklyTimetable.query.count())

# ── DB Init ───────────────────────────────────────────────────────────────

def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin",
                         email="admin@studyplanner.local",
                         password_hash=generate_password_hash("Admin@123"),
                         role="admin")
            db.session.add(admin); db.session.flush()
            db.session.add(Profile(user_id=admin.id, full_name="Administrator"))
            db.session.commit()

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
