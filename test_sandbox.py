
import pandas as pd

from app.sandbox.docker_runner import execute_analysis_code


df = pd.DataFrame({
    "budget": [100, 200, 300, 400],
    "revenue": [150, 260, 390, 510],
})

code = """
result = pd.DataFrame({
    "correlation": [df["budget"].corr(df["revenue"])]
})
"""

output = execute_analysis_code(code, df)

print(output)
