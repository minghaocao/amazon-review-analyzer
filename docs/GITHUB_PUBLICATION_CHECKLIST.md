# GitHub Publication Checklist

Use this checklist immediately before creating or connecting a GitHub repository. Step 17 prepares the local repository only; it does not create a remote or publish files.

## Repository

- [ ] Local Git repository initialized
- [ ] Intended default branch confirmed
- [ ] `.gitignore` reviewed against the actual repository contents
- [ ] Generated runtime state and personal data are ignored
- [ ] Repository size is reasonable
- [ ] Release ZIP inclusion is an explicit owner decision
- [ ] GitHub remote is not yet configured, or its intended owner/repository has been confirmed

## Privacy and Security

- [ ] No API keys, access tokens, passwords or credentials
- [ ] No cookies or Amazon session data
- [ ] No `.env` files; only safe example templates are eligible
- [ ] No private absolute paths such as `/Users/<name>/...`
- [ ] No user review uploads or user-specific analysis outputs
- [ ] No local browser profiles
- [ ] No `.venv`, Python cache or IDE cache directories
- [ ] No Ollama models or other model payloads
- [ ] No runtime PID, lock or current launcher log files

## Public Materials

- [ ] README is complete and uses repository-relative links
- [ ] Product disclaimer is visible
- [ ] Screenshots are curated and free of private information
- [ ] Sample data is synthetic and documented
- [ ] Sample reports are reviewed and described as directional examples
- [ ] Quick Start documentation matches v1.0.1
- [ ] Portfolio case study and summaries are reviewed in the owner's voice
- [ ] License decision is complete, or the lack of a license is made explicit

## Quality and Freeze Controls

- [ ] Application version is 1.0.1
- [ ] Full offline test suite passes
- [ ] Release Health Check passes
- [ ] macOS launcher smoke test passes
- [ ] Localization checks pass
- [ ] Release/package security scans pass
- [ ] Demo Dataset and outputs are unchanged
- [ ] Blind V2, Final Blind, Holdout and Dev V2 assets are unchanged
- [ ] Saved Analyses, historical reports and comparison data are unchanged
- [ ] v1.0.0 historical release is unchanged
- [ ] v1.0.1 release semantics are unchanged

## Final Owner Actions

- [ ] Review `git status` and the complete first-commit diff
- [ ] Decide whether evaluation datasets/results should be public or retained privately
- [ ] Decide whether the packaged `.app` and release ZIP should be tracked or attached to a future GitHub Release
- [ ] Select a license
- [ ] Configure the GitHub remote manually
- [ ] Create and push the first commit only after the repository owner approves the exact public file set
