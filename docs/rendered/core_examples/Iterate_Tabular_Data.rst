Iterate Over Tabular Data
=========================

**Author:** Eric Charles

**Last Run Successfully:** April 26, 2022

This notebook demonstrates three ways to iterate over tabular data:

1. Using the ``tables_io.iteratorNative`` function

2. Using the ``rail.core.data.TableHandle`` data handle object

3. Using the ``rail.core.stage.RailStage`` functionality

.. code:: ipython3

    # Basic imports
    import os
    import rail
    import tables_io
    from rail.core.stage import RailStage
    from rail.core.data import TableHandle

Get access to the RAIL DataStore, and set it to allow us to overwrite
data.

Allowing overwrites will prevent errors when re-running cells in the
notebook.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

Set up the path to the test data.

.. code:: ipython3

    from rail.core.utils import find_rail_file
    pdfs_file = find_rail_file("examples_data/testdata/test_dc2_training_9816.hdf5")

Get access to the data directly, using the DataStore.read_file function.

This will load the entire table from the file we are reading.

.. code:: ipython3

    data = DS.read_file('input', TableHandle, pdfs_file)

.. code:: ipython3

    print(data())


.. parsed-literal::

    OrderedDict([('photometry', OrderedDict([('id', array([8062500000, 8062500062, 8062500124, ..., 8082681636, 8082693813,
           8082707059])), ('mag_err_g_lsst', array([0.00500126, 0.00508365, 0.00505737, ..., 0.01664717, 0.03818999,
           0.05916394], dtype=float32)), ('mag_err_i_lsst', array([0.00500074, 0.00507535, 0.00501555, ..., 0.0153863 , 0.03277681,
           0.04307469], dtype=float32)), ('mag_err_r_lsst', array([0.00500058, 0.00504773, 0.00501542, ..., 0.0122792 , 0.02692565,
           0.03255744], dtype=float32)), ('mag_err_u_lsst', array([0.00504562, 0.00955173, 0.01114765, ..., 0.20123477, 0.7962344 ,
           0.99701214], dtype=float32)), ('mag_err_y_lsst', array([0.00500337, 0.00580441, 0.005063  , ..., 0.0662687 , 0.14290111,
           0.15717329], dtype=float32)), ('mag_err_z_lsst', array([0.0050014 , 0.0051933 , 0.00502286, ..., 0.0272381 , 0.06901625,
           0.07261812], dtype=float32)), ('mag_g_lsst', array([16.960892, 20.709402, 20.437067, ..., 24.11405 , 25.068745,
           25.552408], dtype=float32)), ('mag_i_lsst', array([16.50631 , 20.437565, 19.31263 , ..., 23.711334, 24.587885,
           24.891462], dtype=float32)), ('mag_r_lsst', array([16.653412, 20.533852, 19.709715, ..., 23.828472, 24.770744,
           24.984402], dtype=float32)), ('mag_u_lsst', array([18.040369, 21.61559 , 21.851952, ..., 25.185795, 26.682219,
           26.926563], dtype=float32)), ('mag_y_lsst', array([16.423904, 20.38821 , 18.770441, ..., 23.83491 , 24.673431,
           24.777039], dtype=float32)), ('mag_z_lsst', array([16.466377, 20.408886, 18.953411, ..., 23.75624 , 24.786388,
           24.842054], dtype=float32)), ('redshift', array([0.02043499, 0.01936132, 0.03672067, ..., 2.97927326, 2.98694714,
           2.97646626]))]))])


tables_io.iteratorNative function
---------------------------------

This will open the HDF5 file, and iterate over the file, returning
chunks of data

.. code:: ipython3

    # set up the iterator, and see what type of objec the iterator is
    x = tables_io.iteratorNative(pdfs_file, groupname='photometry', chunk_size=1000)
    print(x)
    for xx in x:
        print(xx[0], xx[1], xx[2]['id'][0])


.. parsed-literal::

    <generator object iterHdf5ToDict at 0x7f45e5fafdf0>
    0 1000 8062500000
    1000 2000 8062643020
    2000 3000 8062942715
    3000 4000 8063364908
    4000 5000 8063677075
    5000 6000 8064196253
    6000 7000 8064664220
    7000 8000 8065297891
    8000 9000 8066223293
    9000 10000 8067729889
    10000 10225 8075587302


rail.core.data.TableHandle data handle object
---------------------------------------------

This will create a TableHandle object that points to the correct file,
which can be use to iterate over that file.

.. code:: ipython3

    th = TableHandle('data', path=pdfs_file)
    x = th.iterator(groupname='photometry', chunk_size=1000)
    print(x)
    for xx in x:
        print(xx[0], xx[1], xx[2]['id'][0])


.. parsed-literal::

    <generator object iterHdf5ToDict at 0x7f45e17cc0b0>
    0 1000 8062500000
    1000 2000 8062643020
    2000 3000 8062942715
    3000 4000 8063364908
    4000 5000 8063677075
    5000 6000 8064196253
    6000 7000 8064664220
    7000 8000 8065297891
    8000 9000 8066223293
    9000 10000 8067729889
    10000 10225 8075587302


rail.core.stage.RailStage functionality
---------------------------------------

This will create a RailStage pipeline stage, which takes as input an
HDF5 file, so the ``input_iterator`` function can be used to iterate
over that file.

.. code:: ipython3

    from rail.core.utilStages import ColumnMapper

.. code:: ipython3

    cm = ColumnMapper.make_stage(input=pdfs_file, chunk_size=1000, hdf5_groupname='photometry', columns=dict(id='bob'))
    x = cm.input_iterator('input')
    for  xx in x:
        print(xx[0], xx[1], xx[2]['id'][0])


.. parsed-literal::

    0 1000 8062500000
    1000 2000 8062643020
    2000 3000 8062942715
    3000 4000 8063364908
    4000 5000 8063677075
    5000 6000 8064196253
    6000 7000 8064664220
    7000 8000 8065297891
    8000 9000 8066223293
    9000 10000 8067729889
    10000 10225 8075587302


