from django.urls import path
from .views import (
    DoctorRegisterView, PatientRegisterView, LoginView, UserProfileView, 
    health_check, PatientProfileUpdateView, DoctorProfileUpdateView,
    PatientCreateSessionView, DoctorJoinSessionView, DoctorEndSessionView,
    ActiveSessionStatusView, ConsultationDetailView,
    PatientSessionListView, DoctorSessionListView
)

urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('auth/register/doctor/', DoctorRegisterView.as_view(), name='register_doctor'),
    path('auth/register/patient/', PatientRegisterView.as_view(), name='register_patient'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/me/', UserProfileView.as_view(), name='user_profile'),
    path('auth/profile/update/', PatientProfileUpdateView.as_view(), name='profile_update'),
    path('auth/doctor-profile/update/', DoctorProfileUpdateView.as_view(), name='doctor_profile_update'),

    path('consultation/create/', PatientCreateSessionView.as_view(), name='create_session'),
    path('consultation/join/', DoctorJoinSessionView.as_view(), name='join_session'),
    path('consultation/end/', DoctorEndSessionView.as_view(), name='end_session'),
    path('consultation/status/', ActiveSessionStatusView.as_view(), name='session_status'),
    path('consultation/detail/<str:code>/', ConsultationDetailView.as_view(), name='consultation_detail'),
    path('consultation/patient/list/', PatientSessionListView.as_view(), name='patient_sessions'),
    path('consultation/doctor/list/', DoctorSessionListView.as_view(), name='doctor_sessions'),
]
