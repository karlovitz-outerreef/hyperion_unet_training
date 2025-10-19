# AWS Credentials - Simple Explanation

## The Two Types of Credentials You Need

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS CREDENTIALS OVERVIEW                      │
└─────────────────────────────────────────────────────────────────┘

1. YOUR LAPTOP                          2. SAGEMAKER TRAINING JOB
   (needs ACCESS KEYS)                     (needs IAM ROLE)

   ┌──────────────┐                       ┌──────────────┐
   │              │  Submit job           │              │
   │  You run:    │ ───────────────────>  │  SageMaker   │
   │  python      │                       │  runs your   │
   │  sagemaker_  │                       │  training    │
   │  train.py    │                       │  code        │
   │              │                       │              │
   └──────────────┘                       └──────────────┘
        │                                        │
        │ Uses                                   │ Uses
        │ Access Keys                            │ IAM Role
        ▼                                        ▼
   ┌──────────────┐                       ┌──────────────┐
   │ IAM User     │                       │ IAM Role     │
   │ sagemaker-   │                       │ SageMaker-   │
   │ developer    │                       │ UNetTraining-│
   │              │                       │ Execution    │
   └──────────────┘                       └──────────────┘
```

---

## What's the Difference?

### Access Keys (for YOUR computer)
- **What:** Two strings (Access Key ID + Secret Access Key)
- **Used by:** Your laptop/local machine
- **Stored in:** `~/.aws/credentials` file
- **Purpose:** Proves YOU are allowed to submit SageMaker jobs, push Docker images, etc.
- **Example:**
  ```
  Access Key ID: AKIAIOSFODNN7EXAMPLE
  Secret Key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  ```

### IAM Role (for SageMaker)
- **What:** A set of permissions (no keys/passwords)
- **Used by:** SageMaker service during training
- **Stored in:** AWS (not on your computer)
- **Purpose:** Proves SAGEMAKER is allowed to read your data from S3, write outputs, etc.
- **Example:**
  ```
  Role ARN: arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole
  ```

---

## Step-by-Step: What to Create

### Step 1: Create IAM Role for SageMaker ⭐
```
AWS Console → IAM → Roles → Create Role
├─ Trusted entity: SageMaker
├─ Permissions: AmazonSageMakerFullAccess + S3 access
└─ Name: SageMaker-UNetTraining-ExecutionRole

Result: Role ARN (copy this!)
```

**🎯 You need this ONCE per project**

### Step 2: Create IAM User (for yourself) 👤
```
AWS Console → IAM → Users → Create User
├─ Name: sagemaker-developer (or your name)
├─ Permissions: SageMaker, ECR, S3 access
└─ Create Access Key → Download CSV

Result: Access Key ID + Secret Access Key
```

**🎯 You need this ONCE per developer**

### Step 3: Configure AWS CLI on Your Computer 💻
```bash
aws configure

Enter:
- Access Key ID: [paste from CSV]
- Secret Access Key: [paste from CSV]
- Region: us-east-1
- Format: json
```

**🎯 You need this ONCE per computer**

---

## How They Work Together

```
┌──────────────────────────────────────────────────────────────┐
│  When you run: python sagemaker_train.py                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 1. AWS CLI checks ~/.aws/credentials  │
        │    for YOUR access keys               │
        │    ✓ Access granted                   │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 2. Script tells SageMaker to use      │
        │    the IAM Role ARN you created       │
        └───────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ 3. SageMaker assumes that role and    │
        │    starts training with those perms   │
        │    ✓ Can read S3                      │
        │    ✓ Can pull Docker image            │
        │    ✓ Can write outputs                │
        └───────────────────────────────────────┘
```

---

## Common Questions

### Q: Why can't I just use one credential?
**A:** AWS separates "who you are" (IAM user/access keys) from "what the service can do" (IAM role). This is more secure.

### Q: Where do I use the Access Key?
**A:** Only in `aws configure` on your laptop. Never hardcode it in scripts!

### Q: Where do I use the Role ARN?
**A:** The `sagemaker_train.py` script uses it automatically. You can set it via:
```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMaker-UNetTraining-ExecutionRole
```

### Q: Is my Secret Access Key stored safely?
**A:** Yes, `aws configure` stores it encrypted in `~/.aws/credentials`. Don't share this file!

### Q: What if I lose my Secret Access Key?
**A:** You can't retrieve it. Create a new access key and deactivate the old one.

### Q: Can I use the same role for multiple projects?
**A:** Yes, but it's better to create separate roles per project for security.

---

## Quick Start Commands

**After you've created both IAM user and role:**

```bash
# 1. Configure your local machine (one time)
aws configure
# Enter your Access Key ID and Secret Access Key

# 2. Set the SageMaker role (optional, script auto-detects)
export SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMaker-VesselSegmentation-ExecutionRole

# 3. Submit training job
python sagemaker_train.py --epochs 50
```

---

## Full Setup Guide

For detailed step-by-step instructions with screenshots, see:
- **IAM_SETUP_GUIDE.md** - Complete walkthrough
- **QUICKSTART_SAGEMAKER.md** - Quick reference

---

## Security Reminder 🔒

✅ **Safe:**
- Storing credentials in `~/.aws/credentials`
- Using IAM roles for services
- Rotating access keys every 90 days

❌ **Unsafe:**
- Committing credentials to Git
- Sharing your Secret Access Key
- Hardcoding credentials in code
- Using root account access keys
