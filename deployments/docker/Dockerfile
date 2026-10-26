# Local mirror of the WACV2027 UPAR Track 2 evaluation container.
#
# Same base image the organizers run your submission in (Codabench Track 2,
# https://www.codabench.org/competitions/18196/, Submission page): fixed Python,
# torch, torchvision, numpy, Pillow versions, and no network access at
# inference time. Use this image to build and smoke-test a submission zip
# exactly the way Codabench will run it, before you upload it.
#
# This image is NOT what you submit. You submit a zip; the ingestion
# program runs your run.py inside a container built from this same base
# image. Anything your model needs beyond what this base image already
# ships must travel inside the zip, not be installed here.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# No system packages, no extra pip installs here: the point of this image
# is to catch "works on my machine" drift against the organizers' exact
# runtime, not to build a more capable one. If your submission needs a
# package beyond torch/torchvision/numpy/Pillow, it has to be vendored
# into the submission zip itself (see the Codabench Submission page) —
# installing it here would hide that requirement
# until the real evaluation run fails.

ENTRYPOINT ["python"]
