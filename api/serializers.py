from rest_framework import serializers
from django.contrib.auth import authenticate
from .models import User, DoctorProfile, PatientProfile

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'email', 'phone', 'first_name', 'last_name', 'role']


class DoctorProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorProfile
        fields = [
            'qualification', 'specialization', 'council', 
            'registration_number', 'registration_date', 'hospital', 
            'bio', 'years_of_experience'
        ]


class PatientProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = PatientProfile
        fields = [
            'date_of_birth', 'height', 'weight', 'blood_group', 
            'emergency_contact_name', 'emergency_contact_phone', 
            'insurance_info', 'allergies'
        ]


class PatientProfileUpdateSerializer(serializers.Serializer):
    """Used for PATCH /api/auth/profile/update/ — all fields optional."""
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name  = serializers.CharField(required=False, allow_blank=True)
    phone      = serializers.CharField(required=False, allow_blank=True)

    date_of_birth          = serializers.DateField(required=False, allow_null=True)
    height                 = serializers.DecimalField(max_digits=5, decimal_places=1, required=False, allow_null=True)
    weight                 = serializers.DecimalField(max_digits=5, decimal_places=1, required=False, allow_null=True)
    blood_group            = serializers.CharField(required=False, allow_blank=True)
    emergency_contact_name = serializers.CharField(required=False, allow_blank=True)
    emergency_contact_phone= serializers.CharField(required=False, allow_blank=True)
    insurance_info         = serializers.CharField(required=False, allow_blank=True)
    allergies              = serializers.JSONField(required=False)


class DoctorProfileUpdateSerializer(serializers.Serializer):
    """Used for PATCH /api/auth/doctor-profile/update/ — all fields optional."""
    first_name          = serializers.CharField(required=False, allow_blank=True)
    last_name           = serializers.CharField(required=False, allow_blank=True)
    phone               = serializers.CharField(required=False, allow_blank=True)
    qualification       = serializers.CharField(required=False, allow_blank=True)
    specialization      = serializers.CharField(required=False, allow_blank=True)
    council             = serializers.CharField(required=False, allow_blank=True)
    registration_number = serializers.CharField(required=False, allow_blank=True)
    registration_date   = serializers.DateField(required=False, allow_null=True)
    hospital            = serializers.CharField(required=False, allow_blank=True)
    bio                 = serializers.CharField(required=False, allow_blank=True)
    years_of_experience = serializers.IntegerField(required=False, allow_null=True)



class DoctorRegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    name = serializers.CharField()
    phone = serializers.CharField(required=False, allow_blank=True)
    
    qualification = serializers.CharField(required=False, allow_blank=True)
    specialization = serializers.CharField(required=False, allow_blank=True)
    council = serializers.CharField(required=False, allow_blank=True)
    registration_number = serializers.CharField(required=False, allow_blank=True)
    registration_date = serializers.DateField(required=False, allow_null=True)
    hospital = serializers.CharField(required=False, allow_blank=True)
    bio = serializers.CharField(required=False, allow_blank=True)
    years_of_experience = serializers.IntegerField(required=False, allow_null=True)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def create(self, validated_data):
        email = validated_data.pop('email')
        password = validated_data.pop('password')
        name = validated_data.pop('name')
        phone = validated_data.get('phone', '')
        
        # Split name into first and last names if possible
        name_parts = name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        # Create user
        user = User.objects.create_user(
            username=email, # django user requires unique username, we use email
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            role=User.Role.DOCTOR
        )

        # Create doctor profile
        DoctorProfile.objects.create(
            user=user,
            qualification=validated_data.get('qualification', ''),
            specialization=validated_data.get('specialization', ''),
            council=validated_data.get('council', ''),
            registration_number=validated_data.get('registration_number', ''),
            registration_date=validated_data.get('registration_date', None),
            hospital=validated_data.get('hospital', ''),
            bio=validated_data.get('bio', ''),
            years_of_experience=validated_data.get('years_of_experience', None),
        )

        return user


class PatientRegisterSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)
    name = serializers.CharField()
    phone = serializers.CharField(required=False, allow_blank=True)
    
    date_of_birth = serializers.DateField(required=False, allow_null=True)
    height = serializers.DecimalField(max_digits=5, decimal_places=1, required=False, allow_null=True)
    weight = serializers.DecimalField(max_digits=5, decimal_places=1, required=False, allow_null=True)
    blood_group = serializers.CharField(required=False, allow_blank=True)
    emergency_contact_name = serializers.CharField(required=False, allow_blank=True)
    emergency_contact_phone = serializers.CharField(required=False, allow_blank=True)
    insurance_info = serializers.CharField(required=False, allow_blank=True)
    allergies = serializers.JSONField(required=False, default=list)

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def create(self, validated_data):
        email = validated_data.pop('email')
        password = validated_data.pop('password')
        name = validated_data.pop('name')
        phone = validated_data.get('phone', '')

        # Split name
        name_parts = name.split(' ', 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ''

        # Create user
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            role=User.Role.PATIENT
        )

        # Create patient profile
        PatientProfile.objects.create(
            user=user,
            date_of_birth=validated_data.get('date_of_birth', None),
            height=validated_data.get('height', None),
            weight=validated_data.get('weight', None),
            blood_group=validated_data.get('blood_group', ''),
            emergency_contact_name=validated_data.get('emergency_contact_name', ''),
            emergency_contact_phone=validated_data.get('emergency_contact_phone', ''),
            insurance_info=validated_data.get('insurance_info', ''),
            allergies=validated_data.get('allergies', [])
        )

        return user


class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')

        if email and password:
            user = authenticate(username=email, password=password)
            if not user:
                raise serializers.ValidationError("Unable to log in with provided credentials.")
            if not user.is_active:
                raise serializers.ValidationError("User is deactivated.")
        else:
            raise serializers.ValidationError("Must include both email and password.")

        data['user'] = user
        return data
