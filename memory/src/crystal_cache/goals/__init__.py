"""Goals (Q109 A, 2026-10-24).

Anthony: "'goals' its just a long standing directive that the model
addresses periodically. And consistently works at the next step of each
goal's plans. We will use a very strong model for digesting a goal and
making a plan."

  planner.py   the frontier tier turns the person's words into a plan
               (summary, success measure, how often to check, steps, the
               apps and acts it needs). Re-run on request ("replan").
  session.py   one check-in: the large tier works the plan's next step
               with a fresh toolset (the goal, watched app items, the
               bank, action items, drafts, sending when permitted).
  runner.py    the pass the app-sync worker runs every minute: plan the
               unplanned, run the due, under the tenant's door.
"""
