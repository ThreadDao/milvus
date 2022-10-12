from utils.util_pymilvus import get_latest_tag

# scale object
# IMAGE_REPOSITORY = "registry.milvus.io/milvus/milvus"  # repository of milvus image
IMAGE_REPOSITORY = "harbor.milvus.io/dockerhub/milvusdb/milvus"
# IMAGE_TAG = get_latest_tag(tag_prefix="2.2.0", tag_latest="2.2.0-latest")  # tag of milvus image
# NAMESPACE = "chaos-testing"  # namespace
NAMESPACE = "chaos-testing"
IF_NOT_PRESENT = "IfNotPresent"  # image pullPolicy IfNotPresent
ALWAYS = "Always"  # image pullPolicy Always
