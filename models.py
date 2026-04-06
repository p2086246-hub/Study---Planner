from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20),  default="user")
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    profile    = db.relationship("Profile",         backref="user", uselist=False, cascade="all, delete-orphan")
    subjects   = db.relationship("Subject",         backref="user", lazy=True,     cascade="all, delete-orphan")
    timetables = db.relationship("WeeklyTimetable", backref="user", lazy=True,     cascade="all, delete-orphan")
    streaks    = db.relationship("Streak",          backref="user", lazy=True,     cascade="all, delete-orphan")
    goals      = db.relationship("DailyGoal",       backref="user", lazy=True,     cascade="all, delete-orphan")


class Profile(db.Model):
    __tablename__ = "profiles"
    id                  = db.Column(db.Integer,    primary_key=True)
    user_id             = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False)
    full_name           = db.Column(db.String(120), default="")
    age                 = db.Column(db.Integer,     default=20)
    education_level     = db.Column(db.String(80),  default="")
    study_goal          = db.Column(db.String(200), default="")
    daily_free_hours    = db.Column(db.Float,       default=0.0)
    study_start_time    = db.Column(db.String(10),  default="")
    study_end_time      = db.Column(db.String(10),  default="")
    concentration_level = db.Column(db.Integer,     default=7)
    # planner mode: 'subject' or 'topic'
    planner_mode        = db.Column(db.String(10),  default="subject")


class Subject(db.Model):
    __tablename__ = "subjects"
    id              = db.Column(db.Integer,    primary_key=True)
    user_id         = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False)
    name            = db.Column(db.String(100), nullable=False)
    difficulty      = db.Column(db.Float,       default=5.0)
    exam_date       = db.Column(db.Date,        nullable=True)
    days_left       = db.Column(db.Integer,     default=30)
    priority_score  = db.Column(db.Float,       default=0.0)
    predicted_hours = db.Column(db.Float,       default=1.0)
    color           = db.Column(db.String(10),  default="#4f46e5")
    created_at      = db.Column(db.DateTime,    default=datetime.utcnow)

    topics = db.relationship("Topic", backref="subject", lazy=True, cascade="all, delete-orphan")


class Topic(db.Model):
    """Sub-topics under a Subject — used in topic-wise planner mode."""
    __tablename__ = "topics"
    id         = db.Column(db.Integer,    primary_key=True)
    subject_id = db.Column(db.Integer,    db.ForeignKey("subjects.id"), nullable=False)
    user_id    = db.Column(db.Integer,    db.ForeignKey("users.id"),    nullable=False)
    name       = db.Column(db.String(150), nullable=False)
    difficulty = db.Column(db.Float,       default=5.0)
    # status: pending / completed / revise
    status     = db.Column(db.String(20),  default="pending")
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)


class WeeklyTimetable(db.Model):
    __tablename__ = "weekly_timetable"
    id           = db.Column(db.Integer,    primary_key=True)
    user_id      = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False)
    week_start   = db.Column(db.Date,       nullable=False)
    day_name     = db.Column(db.String(10), nullable=False)
    time_slot    = db.Column(db.String(30), nullable=False)
    subject_name = db.Column(db.String(100), nullable=False)
    slot_type    = db.Column(db.String(20),  default="Study")
    completed    = db.Column(db.Boolean,    default=False)
    # backlog: True means this slot was rescheduled from a missed day
    is_backlog   = db.Column(db.Boolean,    default=False)
    created_at   = db.Column(db.DateTime,   default=datetime.utcnow)


class Streak(db.Model):
    __tablename__ = "streaks"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date           = db.Column(db.Date,    default=datetime.utcnow)
    all_completed  = db.Column(db.Boolean, default=False)
    current_streak = db.Column(db.Integer, default=0)


class DailyGoal(db.Model):
    """User-created daily checklist goals."""
    __tablename__ = "daily_goals"
    id         = db.Column(db.Integer,    primary_key=True)
    user_id    = db.Column(db.Integer,    db.ForeignKey("users.id"), nullable=False)
    date       = db.Column(db.Date,       default=datetime.utcnow)
    text       = db.Column(db.String(200), nullable=False)
    completed  = db.Column(db.Boolean,    default=False)
    created_at = db.Column(db.DateTime,   default=datetime.utcnow)


class ContactMessage(db.Model):
    __tablename__ = "contact_messages"
    id         = db.Column(db.Integer,    primary_key=True)
    name       = db.Column(db.String(120), nullable=False)
    email      = db.Column(db.String(120), nullable=False)
    subject    = db.Column(db.String(200), default="")
    message    = db.Column(db.Text,        nullable=False)
    is_read    = db.Column(db.Boolean,     default=False)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)
