
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

langfuse = get_client()

with langfuse.start_as_current_observation(
    as_type="span",
    name="uitbot-langfuse-connection-test",
    input={"message": "Testing UITBOT Langfuse integration"},
) as span:
    span.update(
        output={"status": "success"}
    )

langfuse.flush()

print("Đã gửi trace kiểm tra lên Langfuse.")
