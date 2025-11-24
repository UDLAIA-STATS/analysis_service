from rest_framework import serializers
from urllib.parse import urlparse

class VideoAnalyzerSerializer(serializers.Serializer):
    video_url = serializers.URLField(required=True)

    class Meta:
        fields = ['video_url']

    def validate_video_url(self, value):
        """
        Valida que la URL del video:
        - Sea una URL válida (automático por URLField)
        - Corresponda a Cloudflare R2 (endpoint)
        - Tenga una extensión de video válida
        """

        parsed = urlparse(value)

        allowed_domains = [
            "r2.cloudflarestorage.com",
            "cloudflare-r2.com",
        ]

        if not any(domain in parsed.netloc for domain in allowed_domains):
            raise serializers.ValidationError(
                "El enlace no pertenece a un dominio válido de Cloudflare R2."
            )
        
        valid_extensions = [".mp4", ".mov", ".avi", ".mkv"]

        if not any(parsed.path.lower().endswith(ext) for ext in valid_extensions):
            raise serializers.ValidationError(
                "El archivo debe ser un video en formato MP4, MOV, AVI o MKV."
            )

        return value
