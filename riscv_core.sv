`ifdef RISCV_FORMAL
  `define RVFI
`endif

// `include "prim_assert.sv"
// `include "dv_fcov_macros.svh"

/**
 * Top level module of the ibex RISC-V core
 */

module ibex_core import ibex_pkg::*; #(
  parameter bit                     PMPEnable        = 1'b0,
  parameter int unsigned            PMPGranularity   = 0,
  parameter int unsigned            PMPNumRegions    = 4,
  parameter ibex_pkg::pmp_cfg_t     PMPRstCfg[16]    = ibex_pkg::PmpCfgRst,
  parameter logic [33:0]            PMPRstAddr[16]   = ibex_pkg::PmpAddrRst,
  parameter ibex_pkg::pmp_mseccfg_t PMPRstMsecCfg    = ibex_pkg::PmpMseccfgRst,
  parameter int unsigned            MHPMCounterNum   = 0,
  parameter int unsigned            MHPMCounterWidth = 40,
  parameter bit                     RV32E            = 1'b0,
  parameter rv32m_e                 RV32M            = RV32MFast,
  parameter rv32b_e                 RV32B            = RV32BNone,
  parameter bit                     BranchTargetALU  = 1'b0,
  parameter bit                     WritebackStage   = 1'b0,
  parameter bit                     ICache           = 1'b0,
  parameter bit                     ICacheECC        = 1'b0,
  parameter int unsigned            BusSizeECC       = BUS_SIZE,
  parameter int unsigned            TagSizeECC       = IC_TAG_SIZE,
  parameter int unsigned            LineSizeECC      = IC_LINE_SIZE,
  parameter bit                     BranchPredictor  = 1'b0,
  parameter bit                     DbgTriggerEn     = 1'b0,
  parameter int unsigned            DbgHwBreakNum    = 1,
  parameter bit                     ResetAll         = 1'b0,
  parameter lfsr_seed_t             RndCnstLfsrSeed  = RndCnstLfsrSeedDefault,
  parameter lfsr_perm_t             RndCnstLfsrPerm  = RndCnstLfsrPermDefault,
  parameter bit                     SecureIbex       = 1'b0,
  parameter bit                     DummyInstructions= 1'b0,
  parameter bit                     RegFileECC       = 1'b0,
  parameter int unsigned            RegFileDataWidth = 32,
  parameter bit                     MemECC           = 1'b0,
  parameter int unsigned            MemDataWidth     = MemECC ? 32 + 7 : 32,
  parameter int unsigned            DmHaltAddr       = 32'h1A110800,
  parameter int unsigned            DmExceptionAddr  = 32'h1A110808
)(
  // Clock and Reset
  input  logic                         clk_i,
  input  logic                         rst_ni,
  input  logic [31:0]                  hart_id_i,
  input  logic [31:0]                  boot_addr_i,

  // Instruction memory interface
  .        instr_req_o,
  input  logic                         instr_gnt_i,
  input  logic                         instr_rvalid_i,
  output logic [31:0]                  instr_addr_o,
  input  logic [MemDataWidth-1:0]      instr_rdata_i,
  input  logic                         instr_err_i,

  // Data memory interface
  .        data_req_o,
  input  logic                         data_gnt_i,
  input  logic                         data_rvalid_i,
  .        data_we_o,
  output logic [3:0]                   data_be_o,
  output logic [31:0]                  data_addr_o,
  output logic [MemDataWidth-1:0]      data_wdata_o,
  input  logic [MemDataWidth-1:0]      data_rdata_i,
  input  logic                         data_err_i,

  // Register file interface
  .        dummy_instr_id_o,
  .        dummy_instr_wb_o,
  output logic [4:0]                   rf_raddr_a_o,
  output logic [4:0]                   rf_raddr_b_o,
  output logic [4:0]                   rf_waddr_wb_o,
  .        rf_we_wb_o,
  output logic [RegFileDataWidth-1:0]  rf_wdata_wb_ecc_o,
  input  logic [RegFileDataWidth-1:0]  rf_rdata_a_ecc_i,
  input  logic [RegFileDataWidth-1:0]  rf_rdata_b_ecc_i,

  // RAMs interface
  output logic [IC_NUM_WAYS-1:0]       ic_tag_req_o,
  .        ic_tag_write_o,
  output logic [IC_INDEX_W-1:0]        ic_tag_addr_o,
  output logic [TagSizeECC-1:0]        ic_tag_wdata_o,
  input  logic [TagSizeECC-1:0]        ic_tag_rdata_i [IC_NUM_WAYS],
  output logic [IC_NUM_WAYS-1:0]       ic_data_req_o,
  .        ic_data_write_o,
  output logic [IC_INDEX_W-1:0]        ic_data_addr_o,
  output logic [LineSizeECC-1:0]       ic_data_wdata_o,
  input  logic [LineSizeECC-1:0]       ic_data_rdata_i [IC_NUM_WAYS],
  input  logic                         ic_scr_key_valid_i,
  .        ic_scr_key_req_o,

  // Interrupt inputs
  input  logic                         irq_software_i,
  input  logic                         irq_timer_i,
  input  logic                         irq_external_i,
  input  logic [14:0]                  irq_fast_i,
  input  logic                         irq_nm_i,       // non-maskeable interrupt
  .        irq_pending_o,

  // Debug Interface
  input  logic                         debug_req_i,
  output crash_dump_t                  crash_dump_o,
  // SEC_CM: EXCEPTION.CTRL_FLOW.LOCAL_ESC
  // SEC_CM: EXCEPTION.CTRL_FLOW.GLOBAL_ESC
  .        double_fault_seen_o,

  // RISC-V Formal Interface
  // Does not comply with the coding standards of _i/_o suffixes, but follows
  // the convention of RISC-V Formal Interface Specification.
`ifdef RVFI
  .        rvfi_valid,
  output logic [63:0]                  rvfi_order,
  output logic [31:0]                  rvfi_insn,
  .        rvfi_trap,
  .        rvfi_halt,
  .        rvfi_intr,
  output logic [ 1:0]                  rvfi_mode,
  output logic [ 1:0]                  rvfi_ixl,
  output logic [ 4:0]                  rvfi_rs1_addr,
  output logic [ 4:0]                  rvfi_rs2_addr,
  output logic [ 4:0]                  rvfi_rs3_addr,
  output logic [31:0]                  rvfi_rs1_rdata,
  output logic [31:0]                  rvfi_rs2_rdata,
  output logic [31:0]                  rvfi_rs3_rdata,
  output logic [ 4:0]                  rvfi_rd_addr,
  output logic [31:0]                  rvfi_rd_wdata,
  output logic [31:0]                  rvfi_pc_rdata,
  output logic [31:0]                  rvfi_pc_wdata,
  output logic [31:0]                  rvfi_mem_addr,
  output logic [ 3:0]                  rvfi_mem_rmask,
  output logic [ 3:0]                  rvfi_mem_wmask,
  output logic [31:0]                  rvfi_mem_rdata,
  output logic [31:0]                  rvfi_mem_wdata,
  output logic [31:0]                  rvfi_ext_pre_mip,
  output logic [31:0]                  rvfi_ext_post_mip,
  .        rvfi_ext_nmi,
  .        rvfi_ext_nmi_int,
  .        rvfi_ext_debug_req,
  .        rvfi_ext_debug_mode,
  .        rvfi_ext_rf_wr_suppress,
  output logic [63:0]                  rvfi_ext_mcycle,
  output logic [31:0]                  rvfi_ext_mhpmcounters [10],
  output logic [31:0]                  rvfi_ext_mhpmcountersh [10],
  .        rvfi_ext_ic_scr_key_valid,
  .        rvfi_ext_irq_valid,
  `endif

  // CPU Control Signals
  // SEC_CM: FETCH.CTRL.LC_GATED
  input  ibex_mubi_t                   fetch_enable_i,
  .        alert_minor_o,
  .        alert_major_internal_o,
  .        alert_major_bus_o,
  output ibex_mubi_t                   core_busy_o
);

  //////////////////////
  // Clock management //
  //////////////////////

  localparam int unsigned PMPNumChan      = 3;
  // SEC_CM: CORE.DATA_REG_SW.SCA
  localparam bit          DataIndTiming     = SecureIbex;
  localparam bit          PCIncrCheck       = SecureIbex;
  localparam bit          ShadowCSR         = 1'b0;

  //////////////////////
  //   Signals List   //
  //////////////////////

  // IF/ID signals
  logic        dummy_instr_id;
  logic        instr_valid_id;
  logic        instr_new_id;
  logic [31:0] instr_rdata_id;                 // Instruction sampled inside IF stage
  logic [31:0] instr_rdata_alu_id;             // Instruction sampled inside IF stage (replicated to
                                               // ease fan-out)
  logic [15:0] instr_rdata_c_id;               // Compressed instruction sampled inside IF stage
  logic        instr_is_compressed_id;
  logic        instr_perf_count_id;
  logic        instr_bp_taken_id;
  logic        instr_fetch_err;                // Bus error on instr fetch
  logic        instr_fetch_err_plus2;          // Instruction error is misaligned
  logic        illegal_c_insn_id;              // Illegal compressed instruction sent to ID stage
  logic [31:0] pc_if;                          // Program counter in IF stage
  logic [31:0] pc_id;                          // Program counter in ID stage
  logic [31:0] pc_wb;                          // Program counter in WB stage
  logic [33:0] imd_val_d_ex[2];                // Intermediate register for multicycle Ops
  logic [33:0] imd_val_q_ex[2];                // Intermediate register for multicycle Ops
  logic [1:0]  imd_val_we_ex;

  logic        data_ind_timing;
  logic        dummy_instr_en;
  logic [2:0]  dummy_instr_mask;
  logic        dummy_instr_seed_en;
  logic [31:0] dummy_instr_seed;
  logic        icache_enable;
  logic        icache_inval;
  logic        icache_ecc_error;
  logic        pc_mismatch_alert;
  logic        csr_shadow_err;

  logic        instr_first_cycle_id;
  logic        instr_valid_clear;
  logic        pc_set;
  logic        nt_branch_mispredict;
  logic [31:0] nt_branch_addr;
  pc_sel_e     pc_mux_id;                      // Mux selector for next PC
  exc_pc_sel_e exc_pc_mux_id;                  // Mux selector for exception PC
  exc_cause_t  exc_cause;                      // Exception cause

  logic        instr_intg_err;
  logic        lsu_load_err, lsu_load_err_raw;
  logic        lsu_store_err, lsu_store_err_raw;
  logic        lsu_load_resp_intg_err;
  logic        lsu_store_resp_intg_err;

  logic        expecting_load_resp_id;
  logic        expecting_store_resp_id;

  // LSU signals
  logic        lsu_addr_incr_req;
  logic [31:0] lsu_addr_last;

  // Jump and branch target and decision (EX->IF)
  logic [31:0] branch_target_ex;
  logic        branch_decision;

  // Core busy signals
  logic        ctrl_busy;
  logic        if_busy;
  logic        lsu_busy;

  // Register File
  logic [4:0]  rf_raddr_a;
  logic [31:0] rf_rdata_a;
  logic [4:0]  rf_raddr_b;
  logic [31:0] rf_rdata_b;
  logic        rf_ren_a;
  logic        rf_ren_b;
  logic [4:0]  rf_waddr_wb;
  logic [31:0] rf_wdata_wb;
  // Writeback register write data that can be used on the forwarding path (doesn't factor in memory
  // read data as this is too late for the forwarding path)
  logic [31:0] rf_wdata_fwd_wb;
  logic [31:0] rf_wdata_lsu;
  logic        rf_we_wb;
  logic        rf_we_lsu;
  logic        rf_ecc_err_comb;

  logic [4:0]  rf_waddr_id;
  logic [31:0] rf_wdata_id;
  logic        rf_we_id;
  logic        rf_rd_a_wb_match;
  logic        rf_rd_b_wb_match;

  // ALU Control
  alu_op_e     alu_operator_ex;
  logic [31:0] alu_operand_a_ex;
  logic [31:0] alu_operand_b_ex;

  logic [31:0] bt_a_operand;
  logic [31:0] bt_b_operand;

  logic [31:0] alu_adder_result_ex;    // Used to forward computed address to LSU
  logic [31:0] result_ex;

  // Multiplier Control
  logic        mult_en_ex;
  logic        div_en_ex;
  logic        mult_sel_ex;
  logic        div_sel_ex;
  md_op_e      multdiv_operator_ex;
  logic [1:0]  multdiv_signed_mode_ex;
  logic [31:0] multdiv_operand_a_ex;
  logic [31:0] multdiv_operand_b_ex;
  logic        multdiv_ready_id;

  // CSR control
  logic        csr_access;
  csr_op_e     csr_op;
  logic        csr_op_en;
  csr_num_e    csr_addr;
  logic [31:0] csr_rdata;
  logic [31:0] csr_wdata;
  logic        illegal_csr_insn_id;    // CSR access to non-existent register,
                                       // with wrong priviledge level,
                                       // or missing write permissions

  // Data Memory Control
  logic        lsu_we;
  logic [1:0]  lsu_type;
  logic        lsu_sign_ext;
  logic        lsu_req;
  logic        lsu_rdata_valid;
  logic [31:0] lsu_wdata;
  logic        lsu_req_done;

  // stall control
  logic        id_in_ready;
  logic        ex_valid;

  logic        lsu_resp_valid;
  logic        lsu_resp_err;

  // Signals between instruction core interface and pipe (if and id stages)
  logic        instr_req_int;          // Id stage asserts a req to instruction core interface
  logic        instr_req_gated;
  logic        instr_exec;

  // Writeback stage
  logic           en_wb;
  wb_instr_type_e instr_type_wb;
  logic           ready_wb;
  logic           rf_write_wb;
  logic           outstanding_load_wb;
  logic           outstanding_store_wb;
  logic           dummy_instr_wb;

  // Interrupts
  logic        nmi_mode;
  irqs_t       irqs;
  logic        csr_mstatus_mie;
  logic [31:0] csr_mepc, csr_depc;

  // PMP signals
  logic [33:0]  csr_pmp_addr [PMPNumRegions];
  pmp_cfg_t     csr_pmp_cfg  [PMPNumRegions];
  pmp_mseccfg_t csr_pmp_mseccfg;
  logic         pmp_req_err  [PMPNumChan];
  logic         data_req_out;

  logic        csr_save_if;
  logic        csr_save_id;
  logic        csr_save_wb;
  logic        csr_restore_mret_id;
  logic        csr_restore_dret_id;
  logic        csr_save_cause;
  logic        csr_mtvec_init;
  logic [31:0] csr_mtvec;
  logic [31:0] csr_mtval;
  logic        csr_mstatus_tw;
  priv_lvl_e   priv_mode_id;
  priv_lvl_e   priv_mode_lsu;

  // debug mode and dcsr configuration
  logic        debug_mode;
  logic        debug_mode_entering;
  dbg_cause_e  debug_cause;
  logic        debug_csr_save;
  logic        debug_single_step;
  logic        debug_ebreakm;
  logic        debug_ebreaku;
  logic        trigger_match;

  // signals relating to instruction movements between pipeline stages
  // used by performance counters and RVFI
  logic        instr_id_done;
  logic        instr_done_wb;

  logic        perf_instr_ret_wb;
  logic        perf_instr_ret_compressed_wb;
  logic        perf_instr_ret_wb_spec;
  logic        perf_instr_ret_compressed_wb_spec;
  logic        perf_iside_wait;
  logic        perf_dside_wait;
  logic        perf_mul_wait;
  logic        perf_div_wait;
  logic        perf_jump;
  logic        perf_branch;
  logic        perf_tbranch;
  logic        perf_load;
  logic        perf_store;

  // for RVFI
  logic        illegal_insn_id, unused_illegal_insn_id; // ID stage sees an illegal instruction

  //////////////////////
  // Clock management //
  //////////////////////

  // Before going to sleep, wait for I- and D-side
  // interfaces to finish ongoing operations.
  if (SecureIbex) begin : g_core_busy_secure
    // For secure Ibex, the individual bits of core_busy_o are generated from different copies of
    // the various busy signal.
    localparam int unsigned NumBusySignals = 3;
    localparam int unsigned NumBusyBits = $bits(ibex_mubi_t) * NumBusySignals;
    logic [NumBusyBits-1:0] busy_bits_buf;
    prim_buf #(
      .Width(NumBusyBits)
    ) u_fetch_enable_buf (
      .in_i ({$bits(ibex_mubi_t){ctrl_busy, if_busy, lsu_busy}}),
      .out_o(busy_bits_buf)
    );

    // Set core_busy_o to IbexMuBiOn if even a single input is high.
    for (genvar i = 0; i < $bits(ibex_mubi_t); i++) begin : g_core_busy_bits
      if (IbexMuBiOn[i] == 1'b1) begin : g_pos
        assign core_busy_o[i] =  |busy_bits_buf[i*NumBusySignals +: NumBusySignals];
      end else begin : g_neg
        assign core_busy_o[i] = ~|busy_bits_buf[i*NumBusySignals +: NumBusySignals];
      end
    end
  end else begin : g_core_busy_non_secure
    // For non secure Ibex, synthesis is allowed to optimize core_busy_o.
    assign core_busy_o = (ctrl_busy || if_busy || lsu_busy) ? IbexMuBiOn : IbexMuBiOff;
  end

  //////////////////////
  //     IF Stage     //
  //////////////////////

  IF_top #(
    .DmHaltAddr       (DmHaltAddr),
    .DmExceptionAddr  (DmExceptionAddr),
    .DummyInstructions(DummyInstructions),
    .ICache           (ICache),
    .ICacheECC        (ICacheECC),
    .BusSizeECC       (BusSizeECC),
    .TagSizeECC       (TagSizeECC),
    .LineSizeECC      (LineSizeECC),
    .PCIncrCheck      (PCIncrCheck),
    .ResetAll         (ResetAll),
    .RndCnstLfsrSeed  (RndCnstLfsrSeed),
    .RndCnstLfsrPerm  (RndCnstLfsrPerm),
    .BranchPredictor  (BranchPredictor),
    .MemECC           (MemECC),
    .MemDataWidth     (MemDataWidth)
  ) if_stage_i (
    .clk_i (clk_i),
    .rst_ni(rst_ni),

    .boot_addr_i(boot_addr_i),      // input core
    .req_i      (instr_req_gated),  // assign ở ngoài core vào instruction request control

    // instruction cache interface
    .instr_req_o       (instr_req_o),     // NOT CONNECTED
    .instr_addr_o      (instr_addr_o),    // NOT CONNECTED
    .instr_gnt_i       (instr_gnt_i),     // INPUT CORE
    .instr_rvalid_i    (instr_rvalid_i),  // INPUT CORE
    .instr_rdata_i     (instr_rdata_i),   // INPUT CORE
    .instr_bus_err_i   (instr_err_i),     // INPUT CORE
    .instr_intg_err_o  (instr_intg_err),  // NOT CONNECTED

    // ICache RAM IO
    .ic_tag_req_o      (ic_tag_req_o),        // OUTPUT CORE
    .ic_tag_write_o    (ic_tag_write_o),      // OUTPUT CORE 
    .ic_tag_addr_o     (ic_tag_addr_o),       // OUTPUT CORE
    .ic_tag_wdata_o    (ic_tag_wdata_o),      // OUTPUT CORE
    .ic_tag_rdata_i    (ic_tag_rdata_i),      // INPUT CORE
    .ic_data_req_o     (ic_data_req_o),       // OUTPUT CORE
    .ic_data_write_o   (ic_data_write_o),     // OUTPUT CORE
    .ic_data_addr_o    (ic_data_addr_o),      // OUTPUT CORE
    .ic_data_wdata_o   (ic_data_wdata_o),     // OUTPUT CORE
    .ic_data_rdata_i   (ic_data_rdata_i),     // INPUT CORE
    .ic_scr_key_valid_i(ic_scr_key_valid_i),  // INPUT CORE
    .ic_scr_key_req_o  (ic_scr_key_req_o),    // OUTPUT CORE

    // control signals
    .instr_valid_clear_i   (instr_valid_clear),     // NOT CONNECTED
    .pc_set_i              (pc_set),                // NOT CONNECTED
    .pc_mux_i              (pc_mux_id),             // NOT CONNECTED
    .nt_branch_mispredict_i(nt_branch_mispredict),  // NOT CONNECTED

    .nt_branch_addr_i  (nt_branch_addr),     // NOT CONNECTED     // not taken branch address in ID/EX

    .exc_pc_mux_i          (exc_pc_mux_id),   // NOT CONNECTED
    .exc_cause             (exc_cause),       // NOT CONNECTED

    .dummy_instr_en_i      (dummy_instr_en),      // UNUSED
    .dummy_instr_mask_i    (dummy_instr_mask),    // UNUSED
    .dummy_instr_seed_en_i (dummy_instr_seed_en), // UNUSED
    .dummy_instr_seed_i    (dummy_instr_seed),    // UNUSED

    .icache_enable_i       (icache_enable),       // NOT CONNECTED
    .icache_inval_i        (icache_inval),        // NOT CONNECTED
    .icache_ecc_error_o    (icache_ecc_error),    // NOT CONNECTED

    // branch targets
    .branch_target_ex_i(branch_target_ex),        // NOT CONNECTED

    // CSRs
    .csr_mepc_i      (csr_mepc),        // NOT CONNECTED  // exception return address
    .csr_depc_i      (csr_depc),        // NOT CONNECTED  // debug return address
    .csr_mtvec_i     (csr_mtvec),       // NOT CONNECTED  // trap-vector base address
    .csr_mtvec_init_o(csr_mtvec_init),  // NOT CONNECTED

    // pipeline stalls
    .id_in_ready_i(id_in_ready),    // NOT CONNECTED

    // misc signals
    .pc_mismatch_alert_o(pc_mismatch_alert),    // NOT CONNECTED
    .if_busy_o          (if_busy),    // NOT CONNECTED  (1 trong các điều kiện cho core_busy_o)

    // outputs to ID stage
    .instr_valid_id_o        (instr_valid_id),      // CONNECTED ID  // instr in IF-ID is valid
    .instr_new_id_o          (instr_new_id),        // NOT CONNECTED // instr in IF-ID is new
    .instr_rdata_id_o        (instr_rdata_id),      // CONNECTED ID  // instr for ID stage
    .instr_rdata_alu_id_o    (instr_rdata_alu_id),  // CONNECTED ID  // replicated instr for ID stage to reduce fan-out

    .instr_rdata_c_id_o      (instr_rdata_c_id),        // NOT CONNECTED  // compressed instr for ID stage 
    .instr_is_compressed_id_o(instr_is_compressed_id),  // CONNECTED ID
    .instr_bp_taken_o        (instr_bp_taken_id),       // NOT CONNECTED
    .instr_fetch_err_o       (instr_fetch_err),         // CONNECTED ID
    .instr_fetch_err_plus2_o (instr_fetch_err_plus2),   // NOT CONNECTED
    .illegal_c_insn_id_o     (illegal_c_insn_id),       // CONNECTED ID
    
    .dummy_instr_id_o        (dummy_instr_id),          //UNUSED
    .pc_if_o                 (pc_if),                   // NOT CONNECTED
    .pc_id_o                 (pc_id),                   // CONNECTED ID
    .pmp_err_if_i            (pmp_req_err[PMP_I]),      // NOT CONNECTED
    .pmp_err_if_plus2_i      (pmp_req_err[PMP_I2]),     // NOT CONNECTED
  );

  // Core is waiting for the ISide when ID/EX stage is ready for a new instruction but none are
  // available
  assign perf_iside_wait = id_in_ready & ~instr_valid_id;

  // Multi-bit fetch enable used when SecureIbex == 1. When SecureIbex == 0 only use the bottom-bit
  // of fetch_enable_i. Ensure the multi-bit encoding has the bottom bit set for on and unset for
  // off so IbexMuBiOn/IbexMuBiOff can be used without needing to know the value of SecureIbex.

  // `ASSERT_INIT(IbexMuBiSecureOnBottomBitSet,    IbexMuBiOn[0] == 1'b1)
  // `ASSERT_INIT(IbexMuBiSecureOffBottomBitClear, IbexMuBiOff[0] == 1'b0)

  // fetch_enable_i can be used to stop the core fetching new instructions
  if (SecureIbex) begin : g_instr_req_gated_secure
    // For secure Ibex fetch_enable_i must be a specific multi-bit pattern to enable instruction
    // fetch
    // SEC_CM: FETCH.CTRL.LC_GATED
    assign instr_req_gated = instr_req_int & (fetch_enable_i == IbexMuBiOn);
    assign instr_exec      = fetch_enable_i == IbexMuBiOn;
  end else begin : g_instr_req_gated_non_secure
    // For non secure Ibex only the bottom bit of fetch enable is considered
    logic unused_fetch_enable;
    assign unused_fetch_enable = ^fetch_enable_i[$bits(ibex_mubi_t)-1:1];

    assign instr_req_gated = instr_req_int & fetch_enable_i[0];
    assign instr_exec      = fetch_enable_i[0];
  end


  //////////////////////
  //     ID Stage     //
  //////////////////////

  ID_top #(
    .RV32E          (RV32E),
    .RV32M          (RV32M),
    .RV32B          (RV32B),
    .BranchTargetALU(BranchTargetALU),
    .DataIndTiming  (DataIndTiming),
    .WritebackStage (WritebackStage),
    .BranchPredictor(BranchPredictor),
    .MemECC         (MemECC)
  ) id_stage_i (
    .clk_i (clk_i),
    .rst_ni(rst_ni),

    // Interface to IF stage
    .instr_valid_i          (instr_valid_id),
    .instr_fetch_err_i      (instr_fetch_err),
    .instr_rdata_i          (instr_rdata_id),           // from IF-ID pipeline registers
    .instr_rdata_alu_i      (instr_rdata_alu_id),       // from IF-ID pipeline registers
    .instr_is_compressed_i  (instr_is_compressed_id),
    .illegal_c_insn_i       (illegal_c_insn_id),
    .pc_id_i                (pc_id),

    // Branch
    .instr_first_cycle_i     (),      // From FSM   // NOT DONE input logic
    .branch_taken_i          (),      // From FSM   // NOT DONE input logic

    // LSU  Interface
    .lsu_req_EX           (lsu_req),  // to load store unit
    .lsu_we_EX            (lsu_we),  // to load store unit
    .lsu_type_EX          (lsu_type),  // to load store unit
    .lsu_sign_ext_EX      (lsu_sign_ext),  // to load store unit

    // MUL, DIV Interface
    .mult_en_EX            (mult_en_ex),
    .div_en_EX             (div_en_ex),
    .mult_sel_EX           (mult_sel_ex),
    .div_sel_EX            (div_sel_ex),
    .multdiv_operator_EX   (multdiv_operator_ex),
    .multdiv_signed_mode_EX(multdiv_signed_mode_ex),

    // CSR
    .csr_access_EX(csr_access),
    .csr_op_EX(csr_op),
    .csr_op_en_EX(csr_op_en),

    // REG_FILE
    // read
    .rf_raddr_a_o      (rf_raddr_a),
    .rf_rdata_a_i      (rf_rdata_a),
    .rf_raddr_b_o      (rf_raddr_b),
    .rf_rdata_b_i      (rf_rdata_b),
    .rf_ren_a_o        (rf_ren_a),
    .rf_ren_b_o        (rf_ren_b),

    .rf_rdata_a_EX      (),       // NOT DONE output logic [31:0] 
    .rf_rdata_b_EX      (),       // NOT DONE output logic [31:0] 
    // write
    .rf_wdata_sel_EX    (),       // NOT DONE output ibex_pkg::rf_wd_sel_e
    .rf_we_EX           (),       // NOT DONE output logic
    .rf_waddr_EX        (),       // NOT DONE output logic [4:0]

    // IMM
    .imm_a_mux_sel_EX   (),
    .imm_b_mux_sel_EX   (),

    .instr_EX               (),   // NOT DONE output [31:0]
    .instr_rs1_EX           (),   // NOT DONE output [4:0]
    .instr_rs2_EX           (),   // NOT DONE output [4:0]
    .instr_rs3_EX           (),   // NOT DONE output [4:0]
    .instr_rd_EX            (),   // NOT DONE output [4:0]
    .pc_EX                  (),   // NOT DONE output [31:0]
    .instr_is_compressed_EX (),   // NOT DONE output logic
      
    .imm_i_type_EX          (),   // NOT DONE output [31:0]
    .imm_b_type_EX          (),   // NOT DONE output [31:0]
    .imm_s_type_EX          (),   // NOT DONE output [31:0]
    .imm_j_type_EX          (),   // NOT DONE output [31:0]
    .imm_u_type_EX          (),   // NOT DONE output [31:0]
    .zimm_rs1_type_EX       (),   // NOT DONE output [31:0]
    
    // BTALU
    .bt_a_mux_sel_EX        (),   // NOT DONE output op_a_sel_e
    .bt_b_mux_sel_EX        (),   // NOT DONE output imm_b_sel_e
      
    // ALU
    .alu_operator_EX      (),     // NOT DONE output   alu_op_e
    .alu_op_a_mux_sel_EX  (),     // NOT DONE output   op_a_sel_e
    .alu_op_b_mux_sel_EX  (),     // NOT DONE output   op_b_sel_e
    
    // CONTROLLER INTERFACE
    .illegal_insn_o   (),     // NOT DONE output logic
    .ebrk_insn_o      (),     // NOT DONE output logic
    .mret_insn_o      (),     // NOT DONE output logic
                                              
    .dret_insn_o      (),     // NOT DONE output logic
    .ecall_insn_o     (),     // NOT DONE output logic
    .wfi_insn_o       (),     // NOT DONE output logic
    .jump_set_o       (),     // NOT DONE output logic
    .icache_inval_o   ()      // NOT DONE output logic
  );


//////////////////////////
///////// CONTROLLER //////
///////////////////////////

Controller #(
    .WritebackStage (WritebackStage),
    .BranchPredictor(BranchPredictor),
    .MemECC(MemECC)
  ) controller_i (
    .clk_i (clk_i),
    .rst_ni(rst_ni),

    .ctrl_busy_o(ctrl_busy_o),

    // decoder related signals
    .illegal_insn_i  (illegal_insn_o),
    .ecall_insn_i    (ecall_insn_dec),
    .mret_insn_i     (mret_insn_dec),
    .dret_insn_i     (dret_insn_dec),
    .wfi_insn_i      (wfi_insn_dec),
    .ebrk_insn_i     (ebrk_insn),
    .csr_pipe_flush_i(csr_pipe_flush),

    // from IF-ID pipeline
    .instr_valid_i          (instr_valid_i),
    .instr_i                (instr_rdata_i),
    .instr_compressed_i     (instr_rdata_c_i),
    .instr_is_compressed_i  (instr_is_compressed_i),
    .instr_bp_taken_i       (instr_bp_taken_i),
    .instr_fetch_err_i      (instr_fetch_err_i),
    .instr_fetch_err_plus2_i(instr_fetch_err_plus2_i),
    .pc_id_i                (pc_id_i),

    // to IF-ID pipeline
    .instr_valid_clear_o(instr_valid_clear_o),
    .id_in_ready_o      (id_in_ready_o),
    .controller_run_o   (controller_run),
    .instr_exec_i       (instr_exec_i),

    // to prefetcher
    .instr_req_o           (instr_req_o),
    .pc_set_o              (pc_set_o),
    .pc_mux_o              (pc_mux_o),
    .nt_branch_mispredict_o(nt_branch_mispredict_o),
    .exc_pc_mux_o          (exc_pc_mux_o),
    .exc_cause_o           (exc_cause_o),

    // LSU
    .lsu_addr_last_i    (lsu_addr_last_i),
    .load_err_i         (lsu_load_err_i),
    .mem_resp_intg_err_i(mem_resp_intg_err),
    .store_err_i        (lsu_store_err_i),
    .wb_exception_o     (wb_exception),
    .id_exception_o     (id_exception),

    // jump/branch control
    .branch_set_i     (branch_set),
    .branch_not_set_i (branch_not_set),
    .jump_set_i       (jump_set),

    // interrupt signals
    .csr_mstatus_mie_i(csr_mstatus_mie_i),
    .irq_pending_i    (irq_pending_i),
    .irqs_i           (irqs_i),
    .irq_nm_ext_i     (irq_nm_i),
    .nmi_mode_o       (nmi_mode_o),

    // CSR Controller Signals
    .csr_save_if_o        (csr_save_if_o),
    .csr_save_id_o        (csr_save_id_o),
    .csr_save_wb_o        (csr_save_wb_o),
    .csr_restore_mret_id_o(csr_restore_mret_id_o),
    .csr_restore_dret_id_o(csr_restore_dret_id_o),
    .csr_save_cause_o     (csr_save_cause_o),
    .csr_mtval_o          (csr_mtval_o),
    .priv_mode_i          (priv_mode_i),

    // Debug Signal
    .debug_mode_o         (debug_mode_o),
    .debug_mode_entering_o(debug_mode_entering_o),
    .debug_cause_o        (debug_cause_o),
    .debug_csr_save_o     (debug_csr_save_o),
    .debug_req_i          (debug_req_i),
    .debug_single_step_i  (debug_single_step_i),
    .debug_ebreakm_i      (debug_ebreakm_i),
    .debug_ebreaku_i      (debug_ebreaku_i),
    .trigger_match_i      (trigger_match_i),

    .stall_id_i(stall_id),
    .stall_wb_i(stall_wb),
    .flush_id_o(flush_id),
    .ready_wb_i(ready_wb_i),

    // Performance Counters
    .perf_jump_o   (perf_jump_o),
    .perf_tbranch_o(perf_tbranch_o)
  );

 // assign multdiv_en_dec   = mult_en_dec | div_en_dec;  // Move sang EX



// instr_excuting is hidden for now becuz it is related to multi cycle // 


  //assign lsu_req         = instr_executing ? data_req_allowed & lsu_req_dec  : 1'b0;
 // assign mult_en_id      = instr_executing ? mult_en_dec                     : 1'b0; // Sang EX
// assign div_en_id       = instr_executing ? div_en_dec                      : 1'b0; // Sang EX

 // assign lsu_req_o               = lsu_req;
  assign lsu_we_o                = lsu_we;           // Signal in Data Mem Control
  assign lsu_type_o              = lsu_type;        // Signal in Data Meme Control
  assign lsu_sign_ext_o          = lsu_sign_ext;    // Signal in Data Mem Control
  //assign lsu_wdata_o             = rf_rdata_b_fwd;  // **** Signal in Forwarding - Comment for later

  // csr_op_en_o is set when CSR access should actually happen.
  // csv_access_o is set when CSR access instruction is present and is used to compute whether a CSR
  // access is illegal. A combinational loop would be created if csr_op_en_o was used along (as
  // asserting it for an illegal csr access would result in a flush that would need to deassert it).
  
  //assign csr_op_en_o             = csr_access_o & instr_executing & instr_id_done_o; 

  assign alu_operator_ex_o           = alu_operator_ex;      //  Assign to an Instance of a typedef called alu_op_e  
                                                             // in ibex_pkg.sv
  assign alu_operand_a_ex_o          = alu_operand_a_ex;     // opA in Ex stage - ALU Control
  assign alu_operand_b_ex_o          = alu_operand_b_ex;     // opB in Ex Stage - ALU Control 

  //assign mult_en_ex_o                = mult_en_id;
 // assign div_en_ex_o                 = div_en_id;

  // assign multdiv_operator_ex_o       = multdiv_operator;
  // assign multdiv_signed_mode_ex_o    = multdiv_signed_mode;
  // assign multdiv_operand_a_ex_o      = rf_rdata_a_fwd;
  // assign multdiv_operand_b_ex_o      = rf_rdata_b_fwd;
endmodule
