# How to Deploy the Application
Deployments to production occur automatically via GitHub Actions when code merges into the main branch. Rollbacks can be executed using the command 'make deploy-rollback' or manually via the AWS ECS dashboard. Always verify the health endpoint at /health after deployment.
