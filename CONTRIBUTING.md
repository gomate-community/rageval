Contributing to RAGEval
----------

> Note: RAGEval is developed under Python 3.8.18

Welcome! RAGEval is a community project that aims to evaluate different modules of RAG system, including query rewriting, document ranking, information compression, evidence verify, answer generating, and result validating. Your experience and what you can contribute are important to the project's success.

Discussion
----------

If you've run into behavior in RAGEval you don't understand, or you're having trouble working out a good way to apply it to your code, or you've found a bug or would like a feature it doesn't have, we want to hear from you!

Our main forum for discussion is the project's [GitHub issue tracker](https://github.com/gomate-community/rageval/issues).  This is the right place to start a discussion of any of the above or most any other topic concerning the project.

First Time Contributors
-----------------------

RAGEval appreciates your contribution! If you are interested in helping improve RAGEval, there are several ways to get started:

* Work on [new metrics and datasets](https://github.com/gomate-community/rageval/tree/main/rageval).
* Try to answer questions on [the issue tracker](https://github.com/gomate-community/rageval/issues).

Submitting Changes
------------------

Even more excellent than a good bug report is a fix for a bug, or the implementation of a much-needed new metrics or benchmarks. 

(*)  We'd love to have your contributions.

(*) If your new feature will be a lot of work, we recommend talking to us early -- see below.

We use the usual GitHub pull-request flow, which may be familiar to you if you've contributed to other projects on GitHub -- see blow. 

Anyone interested in RAGEval may review your code.  One of the RAGEval core developers will merge your pull request when they think it's ready.
For every pull request, we aim to promptly either merge it or say why it's not yet ready; if you go a few days without a reply, please feel
free to ping the thread by adding a new comment.

For a list of RAGEval core developers, see [Readme](https://github.com/gomate-community/rageval/blob/main/README.md).

Contributing Flow
------------------

1. Fork the latest version of [RAGEval](https://github.com/gomate-community/rageval) into your repo.
2. Create an issue under [gomate-Community/rageval](https://github.com/gomate-community/rageval/issues), write description about the bug/enhancement.
3. Clone your forked RAGEval into your machine, add your changes together with associated tests.
4. Run `make test` with terminal, ensure all unit tests & integration tests passed on your computer.
5. Push to your forked repo, then send the pull request to the official repo. In pull request, you need to create a link to the issue you created using `#[issue_id]`, and describe what has been changed.
6. Wait [Codecov](https://app.codecov.io/gh/gomate-community/rageval) generate the coverage report.
7. We'll assign reviewers to review your code.


Your PR will be merged if:
- Funcitonally benefit for the project.
- Passed Countinuous Integration (all unit tests, integration tests and [PEP8](https://www.python.org/dev/peps/pep-0008/) check passed).
- Test coverage didn't decreased, we use [pytest](https://docs.pytest.org/en/latest/).
- With proper docstrings, see codebase as examples.
- With type hints, see [typing](https://docs.python.org/3/library/typing.html). 
- All reviewers approved your changes.


**Thanks and let's improve RAGEval together!**
