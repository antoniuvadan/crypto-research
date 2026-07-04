# Project details
See ./PROJECT_BKG.md for details.

# Rules
- In plan mode, before execution, go into a Question and Answer mode where you
  ask clarifying questions until you have a complete, firm understanding of 
  the requirements

# Code rules
- Abide by good software engineering practices
  - YAGNI - you aren't going to need it: do not write code you do not need.
  - DRY - don't repeat yourself: think carefully about the abstractions you
    use such that you have a high level of code reusability. Avoid abstracting
    unnecessarily at the same time. Be pragmatic.
  - KISS - keep it stupid simple: prefer simplicity at all times.

# Verification
- Once you implement a plan you must write a rigorous test suite containing
  non-trivial tests covering unit tests and integration of the logic you just
  built. Once tests are written, create a seaprate adversarial review agent
  that inspects the test runs and provides a brutally honest review of the test
  results and the code written initially. The adversarial agent should provide
  a report to you, which you should review carefully and update your initial
  code or tests to satisfy the adversarial agent. Go back and forth with the
  review agent until it deems the code and tests satisfactory.

# Rigor
- Be statistically rigorous in all analyses and all of your communication.
- Always report Sharpe ratio and maximum drawdown for alphas.

