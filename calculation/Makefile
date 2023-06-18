FFTW_DIR = /home/moriwaki/.local
HDF5DIR = /home/moriwaki/.local
CFITS_DIR = /home/moriwaki/.local

SRCH = para.h proto.h

CC = gcc
HDF5INCL = -I/${HDF5DIR}/include
HDF5LIB = -L/${HDF5DIR}/lib -lhdf5
FFTW_INCL = -I/${FFTW_DIR}/include
FFTW_LIB = -L/${FFTW_DIR}/lib -lfftw3
CFITS_INCL = -I/${CFITS_DIR}/include
CFITS_LIB = -L/${CFITS_DIR}/lib -lcfitsio

#OPTIONS += -DSAVE_PEAK_FILE

CFLAGS = $(HDF5INCL) $(FFTW_INCL) $(CFITS_INCL) $(OPTIONS)
LIBS = -lm $(HDF5LIB) $(FFTW_LIB) $(CFITS_LIB)

CALC_PS = utils.o calc_ps.o
CALC_MEAN = utils.o calc_mean.o
ONE_POINT_PDF = utils.o one_point_pdf.o
PEAK_2D = utils.o peak_2d.o
ALL = calc_ps calc_mean peak_2d one_point_pdf 

all: $(ALL)

calc_ps: $(CALC_PS)
	$(CC) $(CFLAGS) $(CALC_PS) $(LIBS) -o calc_ps
calc_mean: $(CALC_MEAN)
	$(CC) $(CFLAGS) $(CALC_MEAN) $(LIBS) -o calc_mean
one_point_pdf: $(ONE_POINT_PDF)
	$(CC) $(CFLAGS) $(ONE_POINT_PDF) $(LIBS) -o one_point_pdf
peak_2d: $(PEAK_2D)
	$(CC) $(CFLAGS) $(PEAK_2D) $(LIBS) -o peak_2d

$(CALC_PS): $(SRCH)
$(CALC_CPS): $(SRCH)
$(CALC_MEAN): $(SRCH)
$(ONE_POINT_PDF): $(SRCH)
$(PEAK_2D): $(SRCH)

clean:
	rm -f $(ALL)
	rm -f *.o

.c.o:
	$(CC) $(CFLAGS) -c $<

