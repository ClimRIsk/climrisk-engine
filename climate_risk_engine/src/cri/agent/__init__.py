"""
agent — Job orchestration layer for autonomous climate risk assessment.

Accepts a company name and assessment scope, runs the full pipeline
(entity resolution → data acquisition → risk assessment → trajectory),
and returns a job ID with streaming partial results.
"""
