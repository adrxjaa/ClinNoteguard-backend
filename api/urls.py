from django.urls import path
from .views import (
    DoctorRegisterView, PatientRegisterView, LoginView, UserProfileView, 
    health_check, PatientProfileUpdateView, DoctorProfileUpdateView,
    PatientCreateSessionView, DoctorJoinSessionView, DoctorEndSessionView,
    ActiveSessionStatusView, ConsultationDetailView,
    PatientSessionListView, DoctorSessionListView, ExtractSessionClaimsView,
    DoctorClinicSlotView, DoctorClinicSlotDeleteView, AppointmentBookingView,
    PatientBriefingView, SessionActionLogView, FinalizePrescriptionView,
    PharmacyLookupView,
)

urlpatterns = [
    path('health/', health_check, name='health_check'),
    path('auth/register/doctor/', DoctorRegisterView.as_view(), name='register_doctor'),
    path('auth/register/patient/', PatientRegisterView.as_view(), name='register_patient'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/me/', UserProfileView.as_view(), name='user_profile'),
    path('auth/profile/update/', PatientProfileUpdateView.as_view(), name='profile_update'),
    path('auth/doctor-profile/update/', DoctorProfileUpdateView.as_view(), name='doctor_profile_update'),

    path('clinics/', DoctorClinicSlotView.as_view(), name='clinic_slots'),
    path('clinics/<int:slot_id>/', DoctorClinicSlotDeleteView.as_view(), name='clinic_slot_delete'),
    path('appointments/', AppointmentBookingView.as_view(), name='appointments'),

    path('consultation/create/', PatientCreateSessionView.as_view(), name='create_session'),
    path('consultation/join/', DoctorJoinSessionView.as_view(), name='join_session'),
    path('consultation/end/', DoctorEndSessionView.as_view(), name='end_session'),
    path('consultation/status/', ActiveSessionStatusView.as_view(), name='session_status'),
    path('consultation/detail/<str:code>/', ConsultationDetailView.as_view(), name='consultation_detail'),
    path('consultation/briefing/<str:code>/', PatientBriefingView.as_view(), name='patient_briefing'),
    path('consultation/action/<str:code>/', SessionActionLogView.as_view(), name='session_action_log'),
    path('consultation/extract-claims/<str:code>/', ExtractSessionClaimsView.as_view(), name='extract_claims'),
    path('consultation/prescribe/<str:code>/', FinalizePrescriptionView.as_view(), name='finalize_prescription'),
    path('consultation/patient/list/', PatientSessionListView.as_view(), name='patient_sessions'),
    path('consultation/doctor/list/', DoctorSessionListView.as_view(), name='doctor_sessions'),

    path('pharmacy/lookup/<str:rx_token>/', PharmacyLookupView.as_view(), name='pharmacy_lookup'),
]
