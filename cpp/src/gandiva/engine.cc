// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "engine.h"

#include <arrow/status.h>
#include <bits/types/time_t.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>
#include <stddef.h>
#include <algorithm>
#include <ctime>
#include <system_error>
#include <unordered_map>

#include "arrow.h"

// TODO(wesm): LLVM 7 produces pesky C4244 that disable pragmas around the LLVM
// includes seem to not fix as with LLVM 6
#if defined(_MSC_VER)
#pragma warning(disable : 4244)
#endif

#include "gandiva/engine.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <iomanip>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4141)
#pragma warning(disable : 4146)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4624)
#endif

#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/Transforms/Vectorize.h>

#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Support/TargetRegistry.h>

#include "TCETargetMachine.hh"
#include "TCEStubTargetMachine.hh"
#include "TCETargetMachinePlugin.hh"
#include "Machine.hh"
#include "LLVMBackend.hh"
#include "LLVMTCECmdLineOptions.hh"
#include "InterPassData.hh"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

//For writeTPEF
#include "Program.hh"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "gandiva/configuration.h"
#include "gandiva/decimal_ir.h"
#include "gandiva/exported_funcs_registry.h"

#include "arrow/util/make_unique.h"

// just to be able to manually register tce target if needed.
extern "C" void LLVMInitializeTCETarget();
extern "C" void LLVMInitializeTCETargetInfo();
extern "C" void LLVMInitializeTCEStubTarget();

using namespace llvm;

LLVMBackend *TCEBackend;
TCETargetMachine* targetMachine;
TTAMachine::Machine *target;

namespace gandiva {

extern const unsigned char kPrecompiledBitcode[];
extern const size_t kPrecompiledBitcodeSize;

std::once_flag llvm_init_once_flag;
static bool llvm_init = false;

void Engine::InitOnce() {
  DCHECK_EQ(llvm_init, false);

  // Register target to llvm for using lookupTarget
  LLVMInitializeTCETargetInfo();
  LLVMInitializeTCETarget();
  //LLVMInitializeTCEStubTarget(); //JJH: seems to be needed to set ST which is return for getSubtarget(), but it causes an earlier crash

  //  InitializeAllTargets();
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  InitializeNativeTargetDisassembler();
  sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

  llvm_init = true;
}

Engine::Engine(const std::shared_ptr<Configuration>& conf,
               std::unique_ptr<LLVMContext> ctx,
               std::unique_ptr<ExecutionEngine> engine, Module* module)
    : context_(std::move(ctx)),
      execution_engine_(std::move(engine)),
      ir_builder_(arrow::internal::make_unique<IRBuilder<>>(*context_)),
      module_(module),
      types_(*context_),
      optimize_(conf->optimize()) {}

Status Engine::Init() {
  // Add mappings for functions that can be accessed from LLVM/IR module.
  AddGlobalMappings();

  ARROW_RETURN_NOT_OK(LoadPreCompiledIR());
//  ARROW_RETURN_NOT_OK(DecimalIR::AddFunctions(this)); //JJH: this causes a crash, disable for now but revisit later.

  return Status::OK();
}

/// factory method to construct the engine.
Status Engine::Make(const std::shared_ptr<Configuration>& conf,
                    std::unique_ptr<Engine>* out) {
  std::call_once(llvm_init_once_flag, InitOnce);

  auto ctx = arrow::internal::make_unique<LLVMContext>();
  auto module = arrow::internal::make_unique<Module>("codegen", *ctx);

  // Capture before moving, ExceutionEngine does not allow retrieving the
  // original Module.
  auto module_ptr = module.get();

  //prevent optimizing out the generated code because it is currently not called yet
  const char *argv[] = {"-internalize-public-api-list=_start,_pthread_start,_dthread_start,main"};
  cl::ParseCommandLineOptions(1, argv);

  std::string targetStr ="tcele64";
  std::string errorStr;
  std::string featureString ="";

  std::ofstream LLVMIR_EngineMake_outfile;
  std::time_t time = std::time(nullptr);
  LLVMIR_EngineMake_outfile.open ("LLVMIR_EngineMake_" + std::to_string(time));
  LLVMIR_EngineMake_outfile << "Engine:Make Waiting a short while...\n";
  for (int j = 0; j < 2; j++)
  for (volatile int i = 0; i < 1999999999; i++) ; //Give user some time to attach a debugger
  LLVMIR_EngineMake_outfile << "Engine:Make Continuing... after " << std::time(nullptr) - time;
  LLVMIR_EngineMake_outfile.close();


//  module_ptr->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:64-i16:16:64-i32:32:64-i64:64:64-f32:32:64-f64:64:64-v64:64:64-v128:128:128-v256:256:256-v512:512:512-v1024:1024:1024-a0:0:64-n64");
  module_ptr->setTargetTriple(targetStr);
  // get registered target machine and set plugin.
      const Target* tceTarget =
          TargetRegistry::lookupTarget(targetStr, errorStr);

      if (!tceTarget) {
          std::cerr << "lookupTarget error: " << errorStr << "\n";
      }

      LLVMTCECmdLineOptions* options = new LLVMTCECmdLineOptions;
      Application::setCmdLineOptions(options); //must call before creating LLVMBackend, as that will fetch the options from Application::
      LLVMBackend *ding = new LLVMBackend(false, "/tmp/tcetmpding/");
      target = TTAMachine::Machine::loadFromADF("/home/jjhoozemans/workspaces/TTA/64b_joost.adf");
      std::unique_ptr<TCETargetMachinePlugin> plugin(ding->createPlugin(*target));
      TCEBackend = ding;

      std::string cpuStr = "tce";
      TargetOptions Options;
      targetMachine =
              static_cast<TCETargetMachine*>(
                  tceTarget->createTargetMachine(
                      targetStr, cpuStr, featureString, Options,
                      Reloc::Model::Static));
	  if (!targetMachine) {
		  std::cerr << "Could not create tce target machine" << "\n";
	  }

	  // This hack must be cleaned up before adding TCE target to llvm upstream
	  // these are needed by TCETargetMachine::addInstSelector passes
	  targetMachine->setTargetMachinePlugin(*plugin);
	  targetMachine->setTTAMach(target);
//	  targetMachine->setEmulationModule(emulationModule);

  auto opt_level =
      conf->optimize() ? CodeGenOpt::Aggressive : CodeGenOpt::None;
  // Note that the lifetime of the error string is not captured by the
  // ExecutionEngine but only for the lifetime of the builder. Found by
  // inspecting LLVM sources.
  std::string builder_error;
  std::unique_ptr<ExecutionEngine> exec_engine{
      EngineBuilder(std::move(module))
//          .setMCPU(sys::getHostCPUName())
          .setEngineKind(EngineKind::JIT)
          .setOptLevel(opt_level)
          .setErrorStr(&builder_error)
          .create()};

  if (exec_engine == nullptr) {
    return Status::CodeGenError("Could not instantiate ExecutionEngine: ",
                                builder_error);
  }

  std::unique_ptr<Engine> engine{
      new Engine(conf, std::move(ctx), std::move(exec_engine), module_ptr)};
  ARROW_RETURN_NOT_OK(engine->Init());
  *out = std::move(engine);
  return Status::OK();
}

// Handling for pre-compiled IR libraries.
Status Engine::LoadPreCompiledIR() {
  auto bitcode = StringRef(reinterpret_cast<const char*>(kPrecompiledBitcode),
                                 kPrecompiledBitcodeSize);

  /// Read from file into memory buffer.
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer_or_error =
      MemoryBuffer::getMemBuffer(bitcode, "precompiled", false);

  ARROW_RETURN_IF(!buffer_or_error,
                  Status::CodeGenError("Could not load module from IR: ",
                                       buffer_or_error.getError().message()));

  std::unique_ptr<MemoryBuffer> buffer = move(buffer_or_error.get());

  /// Parse the IR module.
  Expected<std::unique_ptr<Module>> module_or_error =
      getOwningLazyBitcodeModule(move(buffer), *context());
  if (!module_or_error) {
    // NOTE: handleAllErrors() fails linking with RTTI-disabled LLVM builds
    // (ARROW-5148)
    std::string str;
    raw_string_ostream stream(str);
    stream << module_or_error.takeError();
    return Status::CodeGenError(stream.str());
  }
  std::unique_ptr<Module> ir_module = move(module_or_error.get());

  ARROW_RETURN_IF(verifyModule(*ir_module, &errs()),
                  Status::CodeGenError("verify of IR Module failed"));
  ARROW_RETURN_IF(Linker::linkModules(*module_, move(ir_module)),
                  Status::CodeGenError("failed to link IR Modules"));

  return Status::OK();
}

// Get rid of all functions that don't need to be compiled.
// This helps in reducing the overall compilation time. This pass is trivial,
// and is always done since the number of functions in gandiva is very high.
// (Adapted from Apache Impala)
//
// Done by marking all the unused functions as internal, and then, running
// a pass for dead code elimination.
Status Engine::RemoveUnusedFunctions() {
  // Setup an optimiser pipeline
  std::unique_ptr<legacy::PassManager> pass_manager(
      new legacy::PassManager());

  std::unordered_set<std::string> used_functions;
  used_functions.insert(functions_to_compile_.begin(), functions_to_compile_.end());

  pass_manager->add(
      createInternalizePass([&used_functions](const GlobalValue& func) {
        return (used_functions.find(func.getName().str()) != used_functions.end());
      }));
  pass_manager->add(createGlobalDCEPass());
  pass_manager->run(*module_);
  return Status::OK();
}

// Optimise and compile the module.
Status Engine::FinalizeModule() {
  ARROW_RETURN_NOT_OK(RemoveUnusedFunctions());
  std::ofstream preopt_outfile;
  std::time_t time = std::time(nullptr);
  std::string preopt_filename = "LLVMIR_preopt_" + std::to_string(time);
  preopt_outfile.open (preopt_filename);
  preopt_outfile << DumpIR();
  preopt_outfile.close();
  preopt_outfile.flush();
  std::string cmdString = ""
		  "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/tools/LLVM8_tce64/lib "
		  "PATH=/data/tools/LLVM8_tce64/bin:/data/tools/tce64_LLVM8/bin:/data/tools/ghdl/install/bin:$PATH "
		  "/data/tools/LLVM8_tce64/bin/llvm-as " + preopt_filename;
  system(cmdString.c_str());
  InterPassData *ipd = new InterPassData;

  TTAProgram::Program* prog = TCEBackend->compile(
      "/home/jjhoozemans/workspaces/BD_overlay/spark/spark-with-gandiva/" + preopt_filename + ".bc", "",
      *target, 3, true,
      ipd);

  TTAProgram::Program::writeToTPEF(*prog, preopt_filename + ".tpef");

  if (optimize_) {
    // misc passes to allow for inlining, vectorization, ..
    std::unique_ptr<legacy::PassManager> pass_manager(
        new legacy::PassManager());

    TargetIRAnalysis target_analysis =
        execution_engine_->getTargetMachine()->getTargetIRAnalysis();
    pass_manager->add(createTargetTransformInfoWrapperPass(target_analysis));
    pass_manager->add(createFunctionInliningPass());
    pass_manager->add(createInstructionCombiningPass());
    pass_manager->add(createPromoteMemoryToRegisterPass());
    pass_manager->add(createGVNPass());
    pass_manager->add(createNewGVNPass());
    pass_manager->add(createCFGSimplificationPass());
    pass_manager->add(createLoopVectorizePass());
    pass_manager->add(createSLPVectorizerPass());
    pass_manager->add(createGlobalOptimizerPass());

    // run the optimiser
    PassManagerBuilder pass_builder;
    pass_builder.OptLevel = 3;
    pass_builder.populateModulePassManager(*pass_manager);
    pass_manager->run(*module_);
  }

  ARROW_RETURN_IF(verifyModule(*module_, &errs()),
                  Status::CodeGenError("Module verification failed after optimizer"));

  // do the compilation
  execution_engine_->finalizeObject();
  module_finalized_ = true;
  std::ofstream postopt_outfile;
  postopt_outfile.open ("LLVMIR_postopt_" + std::to_string(time));
  postopt_outfile << DumpIR();
  postopt_outfile.close();
  postopt_outfile.flush();


  return Status::OK();
}

void* Engine::CompiledFunction(Function* irFunction) {
  DCHECK(module_finalized_);
  return execution_engine_->getPointerToFunction(irFunction);
}

void Engine::AddGlobalMappingForFunc(const std::string& name, Type* ret_type,
                                     const std::vector<Type*>& args,
                                     void* function_ptr) {
  constexpr bool is_var_arg = false;
  auto prototype = FunctionType::get(ret_type, args, is_var_arg);
  constexpr auto linkage = GlobalValue::ExternalLinkage;
  auto fn = Function::Create(prototype, linkage, name, module());
  execution_engine_->addGlobalMapping(fn, function_ptr);
}

void Engine::AddGlobalMappings() { ExportedFuncsRegistry::AddMappings(this); }

std::string Engine::DumpIR() {
  std::string ir;
  raw_string_ostream stream(ir);
  module_->print(stream, nullptr);
  return ir;
}

}  // namespace gandiva
