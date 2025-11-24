from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from analisis.serializers import VideoAnalyzerSerializer
from analisis.tasks.analysis_runner import run_analysis

class AnalyzeVideoView(APIView):
    def post(self, request):
        serializer = VideoAnalyzerSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        video_url = request.data.get("video_url")
        res = run_analysis(video_url)

        return Response({"message": "Video recibido y validado correctamente.", "result": res})