# SGX private-ANN research framework
#
# Select one implementation at build time:
#   make VARIANT=ours
#   make VARIANT=ohnsw
#   make VARIANT=compass_tee
#
# Convenience targets are also available:
#   make ours
#   make ohnsw
#   make compass_tee
#
# Variant-specific binaries are written to bin/<variant>/. Compatibility
# symlinks are maintained at bin/app, bin/enclave.so, and
# bin/enclave.signed.so so existing evaluation scripts continue to work.

######## Variant selection ########

VARIANT ?= ours
VALID_VARIANTS := ours ohnsw compass_tee

ifeq ($(filter $(VARIANT),$(VALID_VARIANTS)),)
  $(error Unsupported VARIANT '$(VARIANT)'. Choose one of: $(VALID_VARIANTS))
endif

APP_VARIANT_SOURCE     := App/variants/$(VARIANT).cpp
ENCLAVE_VARIANT_SOURCE := Enclave/variants/$(VARIANT).cpp

######## SGX SDK settings ########

SGX_SDK        ?= /opt/intel/sgxsdk
SGX_MODE       ?= HW
SGX_ARCH       ?= x64
SGX_DEBUG      ?= 1
SGX_PRERELEASE ?= 0
SGX_HNSW_MAX_VECTOR_DIM ?= 128

CPPFLAGS += -DSGX_HNSW_MAX_VECTOR_DIM=$(SGX_HNSW_MAX_VECTOR_DIM)

ifeq ($(shell getconf LONG_BIT),32)
  SGX_ARCH := x86
else ifeq ($(findstring -m32,$(CXXFLAGS)),-m32)
  SGX_ARCH := x86
endif

ifeq ($(SGX_ARCH),x86)
  SGX_COMMON_CFLAGS := -m32
  SGX_LIBRARY_PATH  := $(SGX_SDK)/lib
  SGX_ENCLAVE_SIGNER:= $(SGX_SDK)/bin/x86/sgx_sign
  SGX_EDGER8R       := $(SGX_SDK)/bin/x86/sgx_edger8r
else
  SGX_COMMON_CFLAGS := -m64 -maes -msse4.1
  SGX_LIBRARY_PATH  := $(SGX_SDK)/lib64
  SGX_ENCLAVE_SIGNER:= $(SGX_SDK)/bin/x64/sgx_sign
  SGX_EDGER8R       := $(SGX_SDK)/bin/x64/sgx_edger8r
endif

ifeq ($(SGX_DEBUG),1)
  SGX_COMMON_CFLAGS += -O2 -g
else
  SGX_COMMON_CFLAGS += -O2 -D_FORTIFY_SOURCE=2
endif

######## Project paths ########

BIN_ROOT              := bin
VARIANT_BIN_DIR       := $(BIN_ROOT)/$(VARIANT)
APP_PATH              := $(VARIANT_BIN_DIR)/app
ENCLAVE_PATH          := $(VARIANT_BIN_DIR)/enclave.so
SIGNED_ENCLAVE_PATH   := $(VARIANT_BIN_DIR)/enclave.signed.so

ACTIVE_APP_PATH              := $(BIN_ROOT)/app
ACTIVE_ENCLAVE_PATH          := $(BIN_ROOT)/enclave.so
ACTIVE_SIGNED_ENCLAVE_PATH   := $(BIN_ROOT)/enclave.signed.so

ENCLAVE_CONFIG_FILE   := Enclave/Enclave.config.xml
ENCLAVE_SIGNING_KEY   := Enclave/Enclave_private.pem

######## Build mode dependent SGX libraries ########

ifneq ($(SGX_MODE),HW)
  URTS_LIB     := sgx_urts_sim
  UAE_LIB      := sgx_uae_service_sim
  TRTS_LIB     := sgx_trts_sim
  TSERVICE_LIB := sgx_tservice_sim
else
  URTS_LIB     := sgx_urts
  UAE_LIB      := sgx_uae_service
  TRTS_LIB     := sgx_trts
  TSERVICE_LIB := sgx_tservice
endif

TCRYPTO_LIB := sgx_tcrypto

######## Sources ########

APP_COMMON_CPP_FILES := $(wildcard App/*.cpp)
APP_CPP_FILES        := $(APP_COMMON_CPP_FILES) $(APP_VARIANT_SOURCE)
APP_GEN_C_FILE       := App/Enclave_u.c
APP_GEN_H_FILE       := App/Enclave_u.h
APP_GEN_OBJ          := App/Enclave_u.o
APP_OBJS             := $(APP_CPP_FILES:.cpp=.o) $(APP_GEN_OBJ)

ENCLAVE_COMMON_CPP_FILES := $(wildcard Enclave/*.cpp)
ENCLAVE_CPP_FILES        := $(ENCLAVE_COMMON_CPP_FILES) $(ENCLAVE_VARIANT_SOURCE)
ENCLAVE_GEN_C_FILE       := Enclave/Enclave_t.c
ENCLAVE_GEN_H_FILE       := Enclave/Enclave_t.h
ENCLAVE_GEN_OBJ          := Enclave/Enclave_t.o
ENCLAVE_OBJS             := $(ENCLAVE_CPP_FILES:.cpp=.o) $(ENCLAVE_GEN_OBJ)

######## Include paths and flags ########

APP_INCLUDE_PATHS := \
  -IInclude \
  -IApp \
  -IEnclave \
  -I$(SGX_SDK)/include

ENCLAVE_INCLUDE_PATHS := \
  -IInclude \
  -IEnclave \
  -I$(SGX_SDK)/include \
  -I$(SGX_SDK)/include/libcxx \
  -I$(SGX_SDK)/include/tlibc \
  -I$(SGX_SDK)/include/stlport

COMMON_WARNINGS := -Wall -Wextra -Wno-unused-parameter -Wno-unused-function

APP_CFLAGS     := $(SGX_COMMON_CFLAGS) $(COMMON_WARNINGS) -fPIC -Wno-attributes $(APP_INCLUDE_PATHS) $(CPPFLAGS)
APP_CXXFLAGS   := $(APP_CFLAGS) -std=c++17
APP_LDFLAGS    := $(SGX_COMMON_CFLAGS) -L$(SGX_LIBRARY_PATH)
APP_LIBS       := -l$(URTS_LIB) -l$(UAE_LIB) -lssl -lcrypto -lpthread -ldl

ENCLAVE_CFLAGS   := $(SGX_COMMON_CFLAGS) $(COMMON_WARNINGS) -nostdinc -fvisibility=hidden -fpie -fstack-protector $(ENCLAVE_INCLUDE_PATHS) $(CPPFLAGS)
ENCLAVE_CXXFLAGS := $(ENCLAVE_CFLAGS) -std=c++17 -nostdinc++
ENCLAVE_LDFLAGS  := $(SGX_COMMON_CFLAGS) -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles \
                    -L$(SGX_LIBRARY_PATH) \
                    -Wl,--whole-archive -l$(TRTS_LIB) -Wl,--no-whole-archive \
                    -Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l$(TSERVICE_LIB) -l$(TCRYPTO_LIB) -Wl,--end-group \
                    -Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
                    -Wl,-pie,-eenclave_entry -Wl,--export-dynamic \
                    -Wl,--defsym,__ImageBase=0

######## Build mode label ########

ifeq ($(SGX_MODE),HW)
  ifeq ($(SGX_DEBUG),1)
    BUILD_MODE := HW_DEBUG
  else ifeq ($(SGX_PRERELEASE),1)
    BUILD_MODE := HW_PRERELEASE
  else
    BUILD_MODE := HW_RELEASE
  endif
else
  ifeq ($(SGX_DEBUG),1)
    BUILD_MODE := SIM_DEBUG
  else ifeq ($(SGX_PRERELEASE),1)
    BUILD_MODE := SIM_PRERELEASE
  else
    BUILD_MODE := SIM_RELEASE
  endif
endif

######## Targets ########

.PHONY: all activate run clean distclean info ours ohnsw compass_tee

all: activate

ours:
	@$(MAKE) VARIANT=ours all

ohnsw:
	@$(MAKE) VARIANT=ohnsw all

compass_tee:
	@$(MAKE) VARIANT=compass_tee all

info:
	@echo "VARIANT             = $(VARIANT)"
	@echo "VALID_VARIANTS      = $(VALID_VARIANTS)"
	@echo "SGX_SDK             = $(SGX_SDK)"
	@echo "SGX_MODE            = $(SGX_MODE)"
	@echo "SGX_ARCH            = $(SGX_ARCH)"
	@echo "SGX_DEBUG           = $(SGX_DEBUG)"
	@echo "MAX_VECTOR_DIM      = $(SGX_HNSW_MAX_VECTOR_DIM)"
	@echo "BUILD_MODE          = $(BUILD_MODE)"
	@echo "APP_VARIANT_SOURCE  = $(APP_VARIANT_SOURCE)"
	@echo "ENCLAVE_VARIANT_SRC = $(ENCLAVE_VARIANT_SOURCE)"
	@echo "APP_PATH            = $(APP_PATH)"
	@echo "SIGNED_ENCLAVE      = $(SIGNED_ENCLAVE_PATH)"

activate: $(APP_PATH) $(SIGNED_ENCLAVE_PATH)
	@mkdir -p $(BIN_ROOT)
	@ln -sfn "$(VARIANT)/app" $(ACTIVE_APP_PATH)
	@ln -sfn "$(VARIANT)/enclave.so" $(ACTIVE_ENCLAVE_PATH)
	@ln -sfn "$(VARIANT)/enclave.signed.so" $(ACTIVE_SIGNED_ENCLAVE_PATH)
	@printf '%s\n' "$(VARIANT)" > $(BIN_ROOT)/ACTIVE_VARIANT
	@echo "ACTIVE => $(VARIANT)"

run: activate
	@echo "RUN    => $(APP_PATH) [$(BUILD_MODE), $(VARIANT)]"
	@$(APP_PATH) $(SIGNED_ENCLAVE_PATH)

$(VARIANT_BIN_DIR):
	@mkdir -p $@

######## EDL generated files ########

$(APP_GEN_C_FILE) $(APP_GEN_H_FILE): Enclave/Enclave.edl
	@cd App && $(SGX_EDGER8R) --untrusted ../Enclave/Enclave.edl \
		--search-path ../Enclave --search-path ../Include --search-path $(SGX_SDK)/include
	@echo "GEN    => $(APP_GEN_C_FILE)"

$(ENCLAVE_GEN_C_FILE) $(ENCLAVE_GEN_H_FILE): Enclave/Enclave.edl
	@cd Enclave && $(SGX_EDGER8R) --trusted ../Enclave/Enclave.edl \
		--search-path ../Enclave --search-path ../Include --search-path $(SGX_SDK)/include
	@echo "GEN    => $(ENCLAVE_GEN_C_FILE)"

######## App build ########

App/%.o: App/%.cpp $(APP_GEN_H_FILE) Include/shared_types.h
	@$(CXX) $(APP_CXXFLAGS) -c $< -o $@
	@echo "CXX    <= $<"

App/%.o: App/%.c $(APP_GEN_H_FILE)
	@$(CC) $(APP_CFLAGS) -c $< -o $@
	@echo "CC     <= $<"

$(APP_PATH): $(VARIANT_BIN_DIR) $(APP_OBJS)
	@$(CXX) $(APP_OBJS) -o $@ $(APP_LDFLAGS) $(APP_LIBS)
	@echo "LINK   => $@"

######## Enclave build ########

Enclave/%.o: Enclave/%.cpp $(ENCLAVE_GEN_H_FILE) Include/shared_types.h
	@$(CXX) $(ENCLAVE_CXXFLAGS) -c $< -o $@
	@echo "CXX    <= $<"

Enclave/%.o: Enclave/%.c $(ENCLAVE_GEN_H_FILE)
	@$(CC) $(ENCLAVE_CFLAGS) -c $< -o $@
	@echo "CC     <= $<"

$(ENCLAVE_PATH): $(VARIANT_BIN_DIR) $(ENCLAVE_OBJS)
	@$(CXX) $(ENCLAVE_OBJS) -o $@ $(ENCLAVE_LDFLAGS)
	@echo "LINK   => $@"

$(ENCLAVE_SIGNING_KEY):
	@if [ ! -f "$@" ]; then \
		echo "GEN    => $@"; \
		openssl genrsa -3 -out $@ 3072; \
	fi

$(SIGNED_ENCLAVE_PATH): $(ENCLAVE_PATH) $(ENCLAVE_CONFIG_FILE) $(ENCLAVE_SIGNING_KEY)
	@$(SGX_ENCLAVE_SIGNER) sign -key $(ENCLAVE_SIGNING_KEY) -enclave $(ENCLAVE_PATH) -out $@ -config $(ENCLAVE_CONFIG_FILE)
	@echo "SIGN   => $@"

######## Cleanup ########

clean:
	@find App Enclave -type f -name '*.o' -delete
	@rm -f $(APP_GEN_C_FILE) $(APP_GEN_H_FILE) \
	       $(ENCLAVE_GEN_C_FILE) $(ENCLAVE_GEN_H_FILE)
	@rm -rf $(BIN_ROOT)
	@echo "CLEAN  => build products removed"

distclean: clean
	@rm -f $(ENCLAVE_SIGNING_KEY)
	@echo "CLEAN  => generated enclave signing key removed"
