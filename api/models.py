from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    class Role(models.TextChoices):
        DOCTOR = 'DOCTOR', 'Doctor'
        PATIENT = 'PATIENT', 'Patient'

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.PATIENT)
    phone = models.CharField(max_length=20, blank=True, null=True)
    email = models.EmailField(unique=True)

    # Use email as the username for authentication
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return f"{self.email} ({self.role})"


class DoctorProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='doctor_profile')
    qualification = models.CharField(max_length=100, blank=True, null=True)
    specialization = models.CharField(max_length=100, blank=True, null=True)
    council = models.CharField(max_length=100, blank=True, null=True)
    registration_number = models.CharField(max_length=100, blank=True, null=True)
    registration_date = models.DateField(blank=True, null=True)
    hospital = models.CharField(max_length=200, blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    years_of_experience = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"Doctor Profile: {self.user.get_full_name() or self.user.email}"


class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='patient_profile')
    date_of_birth = models.DateField(blank=True, null=True)
    height = models.DecimalField(max_digits=5, decimal_places=1, blank=True, null=True)  # in cm
    weight = models.DecimalField(max_digits=5, decimal_places=1, blank=True, null=True)  # in kg
    blood_group = models.CharField(max_length=10, blank=True, null=True)
    emergency_contact_name = models.CharField(max_length=100, blank=True, null=True)
    emergency_contact_phone = models.CharField(max_length=20, blank=True, null=True)
    insurance_info = models.CharField(max_length=200, blank=True, null=True)
    allergies = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"Patient Profile: {self.user.get_full_name() or self.user.email}"


class ConsultationSession(models.Model):
    class Status(models.TextChoices):
        CREATED = 'CREATED', 'Created'
        CONNECTED = 'CONNECTED', 'Connected'
        ENDED = 'ENDED', 'Ended'

    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patient_sessions')
    doctor = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='doctor_sessions')
    code = models.CharField(max_length=8, unique=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.CREATED)
    created_at = models.DateTimeField(auto_now_add=True)
    transcript_json = models.JSONField(default=list, blank=True)
    soap_note_json = models.JSONField(default=dict, blank=True)
    audio_file = models.FileField(upload_to='consultation_audios/', null=True, blank=True)

    def __str__(self):
        return f"Session {self.code} - {self.status}"

