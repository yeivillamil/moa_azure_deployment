from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id="99e1e721-7184-498e-8aff-b2ad4e53c1c2")
ws = Workspace.create(name='azure-ml',
  subscription_id="eefcc190-8cda-4ad8-a790-c2f984993768",
  resource_group="rg_machine_learnig",
  create_resource_group=False,
  location="eastus2",
  auth=interactive_auth
)
