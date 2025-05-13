# Directory Renaming Considerations

The project has been rebranded from AuditPulse to Bungii, with all application names, titles, and references updated accordingly. The main source code directory is still named `auditpulse_mvp` for continuity.

## If you want to rename the main directory structure:

1. **Create a backup** of your entire project first

2. **Rename directories** from:
   - `auditpulse_mvp` → `bungii`
   - `auditpulse_mvp/frontend` → `bungii/frontend`

3. **Update import statements** throughout the codebase:
   - Search for all imports that include `auditpulse_mvp` and replace with `bungii`
   - Example: `from auditpulse_mvp.core import settings` → `from bungii.core import settings`

4. **Update deployment configurations**:
   - Update `netlify.toml`: change base directory from `auditpulse_mvp/frontend` to `bungii/frontend`
   - Update any CI/CD pipelines to reference the new directory structure

5. **Update documentation**:
   - Update all README files and documentation to reflect the new directory names

6. **Run tests** to ensure everything works with the new structure

This is a more involved process and should be done carefully to avoid breaking the application. 