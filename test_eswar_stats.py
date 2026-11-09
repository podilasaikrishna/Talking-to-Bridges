
import pandas as pd
from analysis.tools import registry

# Create some dummy sensor data (like a vibrating beam)
df = pd.DataFrame({
    'Sensor_1': [1.1, 1.5, 2.9, 1.4, -0.5, 3.3, 1.6, 1.5]
})

# Execute your newly upgraded tool!
result = registry.execute('summary_statistics', {'metric': 'Sensor_1'}, df=df)

import json
print(json.dumps(result, indent=4))

