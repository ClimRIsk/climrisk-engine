"""PRIVATE maintainer tools (git push by the resident agent after authentication). NOT part of the commercial build:
pyproject.toml excludes this package from the wheel, and nothing in the public engine imports it by name (the agent and API load it through
the optional-extension hooks CRI_AGENT_EXTENSIONS and CRI_API_EXTENSIONS, which are set only on your own machine)."""
