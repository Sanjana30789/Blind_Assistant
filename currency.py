from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    print("RESULT:", result)

    if video_frame is not None:
        cv2.imshow("Camera", video_frame.image)
        cv2.waitKey(1)

pipeline = InferencePipeline.init_with_workflow(
    api_key="bWopVNDhPK2NeW9R2UCT",
    workspace_name="ws1-bxj0b",
    workflow_id="detect-count-and-visualize-3",
    video_reference=0,
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()
