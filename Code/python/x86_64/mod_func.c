#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _Gfluct_reg(void);
extern void _etms_reg(void);
extern void _tms_reg(void);
extern void _vecevent_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," Gfluct.mod");
    fprintf(stderr," etms.mod");
    fprintf(stderr," tms.mod");
    fprintf(stderr," vecevent.mod");
    fprintf(stderr, "\n");
  }
  _Gfluct_reg();
  _etms_reg();
  _tms_reg();
  _vecevent_reg();
}
