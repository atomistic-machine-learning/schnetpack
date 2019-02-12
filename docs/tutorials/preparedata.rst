Prepare your data for SchNetPack
================================

Converting xyz-files to extended xyz
------------------------------------

In order to automatically transfer data to an ase database that can be used
to train the model, xyz files must be transformed into extended xyz. Therefore
a comment string with a definition of the atomic properties and the molecular
properties is required. The following snippet is taken from an xyz-file for
`ethanol <http://quantum-machine.org/gdml/>`_ file::

    9
    -97208.40600498248
    C	0.0072	-0.5687	0.0	1.4517297437249999	6.01923561735	5.2067503058e-07
    C	-1.2854	0.2499	0.0	17.9533587683	-5.1623821924	3.49002255005e-07
    O	1.1304	0.3147	0.0	-4.0883694515	22.5901955842	3.30876819905e-06
    H	0.0392	-1.1972	0.89	-1.141572648295	-9.7469181345	7.64734244805
    H	0.0392	-1.1972	-0.89	-1.141572648295	-9.7469181345	-7.64734244805
    H	-1.3175	0.8784	0.89	-2.48205465155	4.933531278999999	4.3699824457
    H	-1.3175	0.8784	-0.89	-2.48205465155	4.933531278999999	-4.3699824457
    H	-2.1422	-0.4239	0.0	-5.5147904611	-3.0206752464999997	-8.9092739811e-09
    H	1.9857	-0.1365	0.0	-2.4392777023	-10.83820307755	-6.07213606025e-08

Usually xyz-files should start with the number of atoms in the first line and
a comment second. In this case the comment line contains the total energy of
the molecule. The following lines contain the atomic properties of the
molecule. In this case the first column contains the atom type, the following
three columns contain the positions and the last three columns consist of the
atomic forces. In order to convert this example to an extended xyz-file the
comment line must store the property information. Therefore the file must be
parsed and converted to the new format. First of all you need to define a
property string. The property string starts with ``Properties=`` and is
followed by teh column information with the shape
``column name:datatype:number of columns``. For the ethanol example above you
get::

    properties = 'Properties=species:S:1:pos:R:3:forces:R:3'

Secondly you need to open a new file and loop over all lines of the xyz-file::

    new_file = open('./new_file.xyz', 'w')
    with open('./old_file', 'r') as xyz_file:
    while True:


The following snippet shows the desired result of a molecular block::

    9
    Properties=species:S:1:pos:R:3:forces:R:3 energy=-97208.40600498248
    C	0.0072	-0.5687	0.0	1.4517297437249999	6.01923561735	5.2067503058e-07
    C	-1.2854	0.2499	0.0	17.9533587683	-5.1623821924	3.49002255005e-07
    O	1.1304	0.3147	0.0	-4.0883694515	22.5901955842	3.30876819905e-06
    H	0.0392	-1.1972	0.89	-1.141572648295	-9.7469181345	7.64734244805
    H	0.0392	-1.1972	-0.89	-1.141572648295	-9.7469181345	-7.64734244805
    H	-1.3175	0.8784	0.89	-2.48205465155	4.933531278999999	4.3699824457
    H	-1.3175	0.8784	-0.89	-2.48205465155	4.933531278999999	-4.3699824457
    H	-2.1422	-0.4239	0.0	-5.5147904611	-3.0206752464999997	-8.9092739811e-09
    H	1.9857	-0.1365	0.0	-2.4392777023	-10.83820307755	-6.07213606025e-08

