"""Evaluation harnesses (experiments as data, runs as artifacts).

Eval-scoped code lives outside the product package on purpose: the product
never imports from here, and nothing here is a product dependency. See
evals/README.md and docs/initiatives/frame-gen-evals/.
"""
