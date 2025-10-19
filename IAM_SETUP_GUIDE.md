# AWS IAM Setup Guide for SageMaker Training

This guide walks you through setting up AWS Identity and Access Management (IAM) for SageMaker training.

## Overview: Two Types of Credentials Needed

1. **SageMaker Execution Role** - Used BY SageMaker during training (no keys needed)
2. **Your IAM User Access Keys** - Used BY YOU to interact with AWS from your computer

---

## Part 1: Create SageMaker Execution Role

This role is what SageMaker assumes when running your training job. It needs permissions to:
- Read training data from S3
- Write model outputs to S3
- Pull Docker images from ECR
- Write logs to CloudWatch

### Step-by-Step Instructions:

1. **Sign in to AWS Console:**
   - Go to https://console.aws.amazon.com/
   - Navigate to **IAM** service

2. **Create the Role:**
   - Click **Roles** (left sidebar) → **Create role**
   - **Trusted entity type:** AWS service
   - **Use case:** SageMaker
   - Select: **SageMaker - Execution**
   - Click **Next**

3. **Attach AWS Managed Policy:**
   - Search for and select: ✅ **AmazonSageMakerFullAccess**
   - Click **Next**

4. **Create Custom S3 Policy:**
   - Click **Create policy** (opens in new tab)
   - Click **JSON** tab
   - Paste this policy:

   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Sid": "S3AccessForVesselSegmentation",
               "Effect": "Allow",
               "Action": [
                   "s3:GetObject",
                   "s3:PutObject",
                   "s3:DeleteObject",
                   "s3:ListBucket"
               ],
               "Resource": [
                   "arn:aws:s3:::vessel-segmentation-data",
                   "arn:aws:s3:::vessel-segmentation-data/*"
               ]
           },
           {
               "Sid": "ECRAccess",
               "Effect": "Allow",
               "Action": [
                   "ecr:GetAuthorizationToken",
                   "ecr:BatchCheckLayerAvailability",
                   "ecr:GetDownloadUrlForLayer",
                   "ecr:BatchGetImage"
               ],
               "Resource": "*"
           },
           {
               "Sid": "CloudWatchLogs",
               "Effect": "Allow",
               "Action": [
                   "logs:CreateLogGroup",
                   "logs:CreateLogStream",
                   "logs:PutLogEvents",
                   "logs:DescribeLogStreams"
               ],
               "Resource": "arn:aws:logs:*:*:*"
           }
       ]
   }
   ```

   - Click **Next: Tags** (skip tags)
   - Click **Next: Review**
   - Policy name: **VesselSegmentationS3ECRAccess**
   - Description: **S3 and ECR access for vessel segmentation training**
   - Click **Create policy**

5. **Return to Role Creation Tab:**
   - Refresh the policies list
   - Search for: **VesselSegmentationS3ECRAccess**
   - ✅ Select it
   - Click **Next**

6. **Name and Create Role:**
   - Role name: **SageMaker-UNetTraining-ExecutionRole**
   - Description: **Execution role for SageMaker UNet training jobs**
   - Review the trust policy (should allow SageMaker service to assume this role)
   - Click **Create role**

7. **Copy the Role ARN:**
   - Click on your newly created role
   - Copy the **ARN** at the top (looks like):
     ```
     arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole
     ```
   - **Save this ARN** - you'll need it!

✅ **SageMaker Execution Role is ready!**

---

## Part 2: Create IAM User with Access Keys

This is for YOUR local machine to interact with AWS (submit jobs, push Docker images, etc.)

### Step-by-Step Instructions:

1. **Create IAM User:**
   - In IAM Console, click **Users** (left sidebar) → **Create user**
   - User name: **sagemaker-developer** (or use your name)
   - Click **Next**

2. **Set Permissions:**
   - Select: **Attach policies directly**
   - Search and select these policies:
     - ✅ **AmazonSageMakerFullAccess**
     - ✅ **AmazonEC2ContainerRegistryFullAccess**
     - ✅ **IAMReadOnlyAccess**
     - ✅ **VesselSegmentationS3ECRAccess** (the one you created earlier)
   - Click **Next**

3. **Review and Create:**
   - Review the permissions
   - Click **Create user**

4. **Create Access Keys:**
   - Click on the user you just created
   - Go to **Security credentials** tab
   - Scroll to **Access keys** section
   - Click **Create access key**

5. **Select Use Case:**
   - Choose: **Command Line Interface (CLI)**
   - ✅ Check: "I understand the above recommendation..."
   - Click **Next**

6. **Set Description (Optional):**
   - Description: "Local development for SageMaker training"
   - Click **Create access key**

7. **📥 DOWNLOAD YOUR CREDENTIALS NOW!**

   You'll see:
   - **Access key ID**: `AKIAIOSFODNN7EXAMPLE`
   - **Secret access key**: `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`

   ⚠️ **CRITICAL:** You can ONLY see the secret key ONCE!

   Options:
   - Click **Download .csv file** (recommended)
   - Or copy both values to a secure location

   Click **Done**

✅ **IAM User with Access Keys is ready!**

---

## Part 3: Configure AWS CLI

Now configure your local machine with the access keys.

```bash
aws configure
```

**Enter your credentials when prompted:**

```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

**Test the configuration:**

```bash
# Verify your identity
aws sts get-caller-identity

# Should output:
# {
#     "UserId": "AIDAI23HXD2O5EXAMPLE",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/sagemaker-developer"
# }

# Test S3 access
aws s3 ls s3://vessel-segmentation-data/

# Test SageMaker access
aws sagemaker list-training-jobs --max-results 5
```

✅ **AWS CLI is configured!**

---

## Part 4: Set Environment Variable for SageMaker Role

You can either:

**Option A: Set environment variable (recommended for development)**

```bash
# Linux/macOS (add to ~/.bashrc or ~/.zshrc)
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole

# Windows PowerShell (add to your PowerShell profile)
$env:SAGEMAKER_ROLE = "arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole"

# Windows Command Prompt
set SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole
```

**Option B: Let the script auto-detect (default)**

The updated `sagemaker_train.py` will automatically use:
```
arn:aws:iam::{your-account-id}:role/SageMaker-UNetTraining-ExecutionRole
```

Just make sure you named your role exactly as instructed above!

---

## Security Best Practices

### ✅ DO:
- Store access keys in `~/.aws/credentials` (done by `aws configure`)
- Use IAM roles instead of access keys whenever possible
- Rotate access keys periodically (every 90 days)
- Use least-privilege permissions
- Enable MFA (Multi-Factor Authentication) on your AWS account

### ❌ DON'T:
- Never commit access keys to Git
- Never share your secret access key
- Don't hardcode credentials in code
- Don't use root account access keys

---

## Troubleshooting

### "Access Denied" when running sagemaker_train.py

**Check 1:** Verify your CLI credentials
```bash
aws sts get-caller-identity
```

**Check 2:** Verify your user has SageMaker permissions
```bash
aws iam list-attached-user-policies --user-name sagemaker-developer
```

**Check 3:** Verify the role exists
```bash
aws iam get-role --role-name SageMaker-UNetTraining-ExecutionRole
```

### "Role not found" error

Make sure you either:
1. Named the role exactly: `SageMaker-UNetTraining-ExecutionRole`
2. OR set `SAGEMAKER_ROLE` environment variable with your actual role ARN

### "Invalid credentials" error

Your access keys may be incorrect. Reconfigure:
```bash
aws configure
```

### Lost your secret access key?

You can't retrieve it. You must:
1. Go to IAM → Users → Your User → Security Credentials
2. **Deactivate** the old access key
3. Create a new access key
4. Run `aws configure` with new credentials

---

## Summary Checklist

- [ ] Created SageMaker Execution Role with policies
- [ ] Copied Role ARN and saved it
- [ ] Created IAM User for local development
- [ ] Created access keys and downloaded CSV
- [ ] Ran `aws configure` with access keys
- [ ] Tested with `aws sts get-caller-identity`
- [ ] Set `SAGEMAKER_ROLE` environment variable (optional)
- [ ] Ready to run `python sagemaker_train.py`!

---

## Quick Reference

**Your IAM Resources:**
```
SageMaker Role ARN: arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole
IAM User: sagemaker-developer
Access Key ID: AKIAIOSFODNN7EXAMPLE (in ~/.aws/credentials)
Region: us-east-1
```

**Next Steps:**
See `QUICKSTART_SAGEMAKER.md` to submit your first training job!
