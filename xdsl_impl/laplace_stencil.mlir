builtin.module {
  func.func private @set_zero() -> f64
  func.func private @ops_par_loop_set_zero_0(%0: !stencil.field<[-1,4097]x[-1,4097]xf64>) {
    %1 = stencil.apply() -> (!stencil.temp<[-1,4097]x[-1,4097]xf64>) {
      %2 = func.call @set_zero() : () -> f64
      stencil.return %2 : f64
    } to <[-1, -1], [4095, 4095]>
    stencil.store %1 to %0(<[-1, -1], [4097, 4097]>) : !stencil.temp<[-1,4097]x[-1,4097]xf64> to !stencil.field<[-1,4097]x[-1,4097]xf64>
    func.return
  }
  func.func private @ops_par_loop_set_zero_1(%0: !stencil.field<[-1,4097]x[-1,4097]xf64>) {
    %1 = stencil.apply() -> (!stencil.temp<[-1,4097]x[-1,4097]xf64>) {
      %2 = func.call @set_zero() : () -> f64
      stencil.return %2 : f64
    } to <[-1, -1], [4095, 4095]>
    stencil.store %1 to %0(<[-1, -1], [4097, 4097]>) : !stencil.temp<[-1,4097]x[-1,4097]xf64> to !stencil.field<[-1,4097]x[-1,4097]xf64>
    func.return
  }
  func.func private @ops_par_loop_set_zero_2(%0: !stencil.field<[-1,4097]x[-1,4097]xf64>) {
    %1 = stencil.apply() -> (!stencil.temp<[-1,4097]x[-1,4097]xf64>) {
      %2 = func.call @set_zero() : () -> f64
      stencil.return %2 : f64
    } to <[-1, -1], [4095, 0]>
    stencil.store %1 to %0(<[-1, -1], [4097, 4097]>) : !stencil.temp<[-1,4097]x[-1,4097]xf64> to !stencil.field<[-1,4097]x[-1,4097]xf64>
    func.return
  }
  func.func private @ops_par_loop_set_zero_3(%0: !stencil.field<[-1,4097]x[-1,4097]xf64>) {
    %1 = stencil.apply() -> (!stencil.temp<[-1,4097]x[-1,4097]xf64>) {
      %2 = func.call @set_zero() : () -> f64
      stencil.return %2 : f64
    } to <[-1, 4094], [4095, 4095]>
    stencil.store %1 to %0(<[-1, -1], [4097, 4097]>) : !stencil.temp<[-1,4097]x[-1,4097]xf64> to !stencil.field<[-1,4097]x[-1,4097]xf64>
    func.return
  }
  func.func private @left_bndcon(index, index) -> f64
  func.func private @ops_par_loop_left_bndcon_4(%0: !stencil.field<[-1,4097]x[-1,4097]xf64>) {
    %1 = stencil.apply() -> (!stencil.temp<[-1,4097]x[-1,4097]xf64>) {
      %2 = stencil.index 0 <[0, 0]>
      %3 = stencil.index 1 <[0, 0]>
      %4 = func.call @left_bndcon(%2, %3) : (index, index) -> f64
      stencil.return %4 : f64
    } to <[-1, -1], [0, 4095]>
    stencil.store %1 to %0(<[-1, -1], [4097, 4097]>) : !stencil.temp<[-1,4097]x[-1,4097]xf64> to !stencil.field<[-1,4097]x[-1,4097]xf64>
    func.return
  }
  func.func private @right_bndcon(index, index) -> f64
  func.func private @ops_par_loop_right_bndcon_5(%0: !stencil.field<[-1,4097]x[-1,4097]xf64>) {
    %1 = stencil.apply() -> (!stencil.temp<[-1,4097]x[-1,4097]xf64>) {
      %2 = stencil.index 0 <[0, 0]>
      %3 = stencil.index 1 <[0, 0]>
      %4 = func.call @right_bndcon(%2, %3) : (index, index) -> f64
      stencil.return %4 : f64
    } to <[4094, -1], [4095, 4095]>
    stencil.store %1 to %0(<[-1, -1], [4097, 4097]>) : !stencil.temp<[-1,4097]x[-1,4097]xf64> to !stencil.field<[-1,4097]x[-1,4097]xf64>
    func.return
  }
}