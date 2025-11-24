from django.urls import path
from analisis.views import AnalyzeVideoView

urlpatterns = [
    path('analyze/', AnalyzeVideoView.as_view(), name='analyze-video'),
]
