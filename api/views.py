from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authtoken.models import Token
from django.http import JsonResponse
from .models import ConsultationSession
from .serializers import (
    DoctorRegisterSerializer,
    PatientRegisterSerializer,
    LoginSerializer,
    UserSerializer,
    DoctorProfileSerializer,
    PatientProfileSerializer,
    PatientProfileUpdateSerializer
)


def health_check(request):
    return JsonResponse({'status': 'ok', 'server': 'clinnote_backend'})

class DoctorRegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = DoctorRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        print(f"--- Doctor Registration Validation Errors: {serializer.errors} ---")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PatientRegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PatientRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user': UserSerializer(user).data
            }, status=status.HTTP_201_CREATED)
        print(f"--- Patient Registration Validation Errors: {serializer.errors} ---")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            token, created = Token.objects.get_or_create(user=user)
            return Response({
                'token': token.key,
                'user': UserSerializer(user).data
            }, status=status.HTTP_200_OK)
        print(f"--- Login Validation Errors: {serializer.errors} ---")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        user_data = UserSerializer(user).data
        
        if user.role == 'DOCTOR':
            profile = getattr(user, 'doctor_profile', None)
            profile_data = DoctorProfileSerializer(profile).data if profile else {}
        else:
            profile = getattr(user, 'patient_profile', None)
            profile_data = PatientProfileSerializer(profile).data if profile else {}

        return Response({
            'user': user_data,
            'profile': profile_data
        }, status=status.HTTP_200_OK)


class PatientProfileUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        user = request.user
        serializer = PatientProfileUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            print(f"--- Profile Update Errors: {serializer.errors} ---")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        # Update User fields
        if 'first_name' in data:
            user.first_name = data['first_name']
        if 'last_name' in data:
            user.last_name = data['last_name']
        if 'phone' in data:
            user.phone = data['phone']
        user.save()

        # Update PatientProfile fields
        profile, _ = user.patient_profile.__class__.objects.get_or_create(user=user)
        for field in ['date_of_birth', 'height', 'weight', 'blood_group',
                      'emergency_contact_name', 'emergency_contact_phone',
                      'insurance_info', 'allergies']:
            if field in data:
                setattr(profile, field, data[field])
        profile.save()

        return Response({
            'user': UserSerializer(user).data,
            'profile': PatientProfileSerializer(profile).data,
        }, status=status.HTTP_200_OK)


class DoctorProfileUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request):
        user = request.user
        from .serializers import DoctorProfileUpdateSerializer
        serializer = DoctorProfileUpdateSerializer(data=request.data)
        if not serializer.is_valid():
            print(f"--- Doctor Profile Update Errors: {serializer.errors} ---")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        data = serializer.validated_data

        # Update User fields
        for field in ['first_name', 'last_name', 'phone']:
            if field in data:
                setattr(user, field, data[field])
        user.save()

        # Update DoctorProfile fields
        from .models import DoctorProfile
        profile, _ = DoctorProfile.objects.get_or_create(user=user)
        for field in ['qualification', 'specialization', 'council', 'registration_number',
                      'registration_date', 'hospital', 'bio', 'years_of_experience']:
            if field in data:
                setattr(profile, field, data[field])
        profile.save()

        return Response({
            'user': UserSerializer(user).data,
            'profile': DoctorProfileSerializer(profile).data,
        }, status=status.HTTP_200_OK)




import random

class PatientCreateSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        if user.role != 'PATIENT':
            return Response({'error': 'Only patients can create a pairing session'}, status=status.HTTP_400_BAD_REQUEST)
        
        # End any existing non-ended sessions for this patient to avoid conflicts
        ConsultationSession.objects.filter(
            patient=user, 
            status__in=[ConsultationSession.Status.CREATED, ConsultationSession.Status.CONNECTED]
        ).update(status=ConsultationSession.Status.ENDED)

        # Generate unique 6-digit code
        while True:
            code = "".join(random.choices("0123456789", k=6))
            if not ConsultationSession.objects.filter(code=code, status__in=['CREATED', 'CONNECTED']).exists():
                break

        session = ConsultationSession.objects.create(
            patient=user,
            code=code,
            status=ConsultationSession.Status.CREATED
        )

        return Response({
            'code': session.code,
            'status': session.status,
        }, status=status.HTTP_201_CREATED)


class DoctorJoinSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        if user.role != 'DOCTOR':
            return Response({'error': 'Only doctors can join a session'}, status=status.HTTP_400_BAD_REQUEST)

        code = request.data.get('code', '').strip()
        try:
            session = ConsultationSession.objects.get(code=code, status=ConsultationSession.Status.CREATED)
        except ConsultationSession.DoesNotExist:
            return Response({'error': 'Active session with this pairing code not found'}, status=status.HTTP_404_NOT_FOUND)

        session.doctor = user
        session.status = ConsultationSession.Status.CONNECTED
        session.save()

        return Response({
            'code': session.code,
            'status': session.status,
            'patient_name': session.patient.get_full_name() or session.patient.email,
        }, status=status.HTTP_200_OK)


class DoctorEndSessionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        code = request.data.get('code', '').strip()
        try:
            session = ConsultationSession.objects.get(code=code, status=ConsultationSession.Status.CONNECTED)
        except ConsultationSession.DoesNotExist:
            return Response({'error': 'Connected session not found'}, status=status.HTTP_404_NOT_FOUND)

        session.status = ConsultationSession.Status.ENDED
        session.save()

        return Response({
            'code': session.code,
            'status': session.status,
        }, status=status.HTTP_200_OK)


class ActiveSessionStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        # Find any active (created/connected) session involving this user
        if user.role == 'PATIENT':
            session = ConsultationSession.objects.filter(
                patient=user, 
                status__in=[ConsultationSession.Status.CREATED, ConsultationSession.Status.CONNECTED]
            ).order_by('-created_at').first()
        else:
            session = ConsultationSession.objects.filter(
                doctor=user, 
                status=ConsultationSession.Status.CONNECTED
            ).order_by('-created_at').first()

        if not session:
            # Check if there is a recently ended session so we can transition to summary page
            ended_session = ConsultationSession.objects.filter(
                patient=user if user.role == 'PATIENT' else None,
                doctor=user if user.role == 'DOCTOR' else None,
                status=ConsultationSession.Status.ENDED
            ).order_by('-created_at').first()
            
            if ended_session:
                return Response({
                    'code': ended_session.code,
                    'status': ended_session.status,
                }, status=status.HTTP_200_OK)

            return Response({'status': 'NONE'}, status=status.HTTP_200_OK)

        return Response({
            'code': session.code,
            'status': session.status,
            'doctor_name': session.doctor.get_full_name() or session.doctor.email if session.doctor else None,
            'patient_name': session.patient.get_full_name() or session.patient.email,
        }, status=status.HTTP_200_OK)


class ConsultationDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, code):
        try:
            session = ConsultationSession.objects.get(code=code)
            return Response({
                'code': session.code,
                'status': session.status,
                'transcript': session.transcript_json,
                'soap_note': session.soap_note_json,
                'doctor_name': session.doctor.get_full_name() or session.doctor.email if session.doctor else None,
                'patient_name': session.patient.get_full_name() or session.patient.email,
            }, status=status.HTTP_200_OK)
        except ConsultationSession.DoesNotExist:
            return Response({'error': 'Consultation not found'}, status=status.HTTP_404_NOT_FOUND)


class PatientSessionListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.role != 'PATIENT':
            return Response({'error': 'Only patients can view patient session list.'}, status=status.HTTP_400_BAD_REQUEST)
        
        sessions = ConsultationSession.objects.filter(
            patient=user,
            status=ConsultationSession.Status.ENDED
        ).order_by('-created_at')
        
        data = []
        for s in sessions:
            data.append({
                'code': s.code,
                'status': s.status,
                'created_at': s.created_at.isoformat(),
                'doctor_name': s.doctor.get_full_name() or s.doctor.email if s.doctor else "Unknown Doctor",
                'patient_name': s.patient.get_full_name() or s.patient.email,
                'soap_note': s.soap_note_json,
            })
            
        return Response(data, status=status.HTTP_200_OK)


class DoctorSessionListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        if user.role != 'DOCTOR':
            return Response({'error': 'Only doctors can view doctor session list.'}, status=status.HTTP_400_BAD_REQUEST)
        
        sessions = ConsultationSession.objects.filter(
            doctor=user,
            status=ConsultationSession.Status.ENDED
        ).order_by('-created_at')
        
        data = []
        for s in sessions:
            data.append({
                'code': s.code,
                'status': s.status,
                'created_at': s.created_at.isoformat(),
                'doctor_name': s.doctor.get_full_name() or s.doctor.email,
                'patient_name': s.patient.get_full_name() or s.patient.email,
                'soap_note': s.soap_note_json,
            })
            
        return Response(data, status=status.HTTP_200_OK)



