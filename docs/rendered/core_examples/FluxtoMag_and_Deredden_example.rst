Flux to Mag And Deredden
========================

author: Sam Schmidt

last successfully run: Apr 26, 2023

.. code:: ipython3

    import matplotlib.pyplot as plt
    import os
    import tables_io
    import tempfile
    from rail.core.stage import RailStage
    from rail.core.data import TableHandle
    from rail.utils.path_utils import find_rail_file
    from rail.tools.photometry_tools import LSSTFluxToMagConverter, Dereddener

.. code:: ipython3

    DS = RailStage.data_store
    example_file = find_rail_file("examples_data/testdata/rubin_dm_dc2_example.pq")
    test_data = DS.read_file("test_data", TableHandle, example_file)


.. parsed-literal::

    column_list None


.. code:: ipython3

    test_data().info()


.. parsed-literal::

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100 entries, 0 to 99
    Data columns (total 15 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   objectId          100 non-null    int64  
     1   ra                100 non-null    float64
     2   decl              100 non-null    float64
     3   u_gaap1p0Flux     100 non-null    float64
     4   u_gaap1p0FluxErr  100 non-null    float64
     5   g_gaap1p0Flux     100 non-null    float64
     6   g_gaap1p0FluxErr  100 non-null    float64
     7   r_gaap1p0Flux     100 non-null    float64
     8   r_gaap1p0FluxErr  100 non-null    float64
     9   i_gaap1p0Flux     100 non-null    float64
     10  i_gaap1p0FluxErr  100 non-null    float64
     11  z_gaap1p0Flux     100 non-null    float64
     12  z_gaap1p0FluxErr  100 non-null    float64
     13  y_gaap1p0Flux     100 non-null    float64
     14  y_gaap1p0FluxErr  100 non-null    float64
    dtypes: float64(14), int64(1)
    memory usage: 11.8 KB


Fluxes to Mags
~~~~~~~~~~~~~~

To convert fluxes to mags, we need to specify patterns for the
``flux_name`` and ``flux_err_name`` columns to be converted, and the
``mag_name`` and ``mag_err_name`` columns that will store the newly
created magnitudes.

This is done as below, by specifying a string listing the bands, and
``{band}`` in the patterns where the individual bands will go. The
dictionary below duplicates the default behavior of the converter, but
is written out explicitly as an example:

.. code:: ipython3

    # convert "gaap" fluxes to magnitudes:
    ftomagdict = dict(bands = "ugrizy",
                      flux_name="{band}_gaap1p0Flux",
                      flux_err_name="{band}_gaap1p0FluxErr",
                      mag_name="mag_{band}_lsst",
                      mag_err_name="mag_err_{band}_lsst",
                      copy_cols=dict(ra='ra', dec='decl', objectId='objectId'))
    fluxToMag = LSSTFluxToMagConverter.make_stage(name='flux2mag', **ftomagdict)

.. code:: ipython3

    mags_data = fluxToMag(test_data)


.. parsed-literal::

    Inserting handle into data store.  output_flux2mag: inprogress_output_flux2mag.hdf5, flux2mag


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/rail/tools/photometry_tools.py:378: RuntimeWarning: invalid value encountered in log10
      return -2.5*np.log10(flux_vals) + self.config.mag_offset


.. code:: ipython3

    list(mags_data().keys())




.. parsed-literal::

    ['mag_u_lsst',
     'mag_err_u_lsst',
     'mag_g_lsst',
     'mag_err_g_lsst',
     'mag_r_lsst',
     'mag_err_r_lsst',
     'mag_i_lsst',
     'mag_err_i_lsst',
     'mag_z_lsst',
     'mag_err_z_lsst',
     'mag_y_lsst',
     'mag_err_y_lsst',
     'ra',
     'dec',
     'objectId']



Deredden Mags
~~~~~~~~~~~~~

To deredden magnitudes we need to grab one of the dust maps used by the
``dustmaps`` package. We’ll grab the default Schlegel-Finkbeiner-Davis
“SFD” map. NOTE: This will download a file to your machine containing
the SFD data!

We need to feed the location of the directory containing the newly
created “sfd” maps to the stage. As we downloaded the data to the
present working directory with the command above, that directory is just
``"./"``

.. code:: ipython3

    dustmap_dir = "./"
    
    dereddener = Dereddener.make_stage(name='dereddener', dustmap_dir=dustmap_dir)
    dereddener.fetch_map()


.. parsed-literal::

    Configuration file not found:
    
        /home/runner/.dustmapsrc
    
    To create a new configuration file in the default location, run the following python code:
    
        from dustmaps.config import config
        config.reset()
    
    Note that this will delete your configuration! For example, if you have specified a data directory, then dustmaps will forget about its location.


.. parsed-literal::

    Downloading SFD data file to /home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_ngp.fits


.. parsed-literal::

    Downloading data to '/home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_ngp.fits' ...
    Downloading https://dataverse.harvard.edu/api/access/datafile/2902687 ...


.. parsed-literal::

      0.0 B of 64.0 MiB |   0.0 s/B |                       | [38;2;255;0;0m  0%[39m | ETA:  --:--:--

.. parsed-literal::

      2.4 MiB of 64.0 MiB |  43.9 MiB/s |                   | [38;2;255;25;0m  3%[39m | ETA:  00:00:00

.. parsed-literal::

      6.5 MiB of 64.0 MiB |  56.9 MiB/s |#                  | [38;2;255;83;0m 10%[39m | ETA:   0:00:01

.. parsed-literal::

     10.5 MiB of 64.0 MiB |  56.9 MiB/s |###                | [38;2;255;101;0m 16%[39m | ETA:   0:00:00

.. parsed-literal::

     14.6 MiB of 64.0 MiB |  63.2 MiB/s |####               | [38;2;255;118;0m 22%[39m | ETA:   0:00:00

.. parsed-literal::

     18.6 MiB of 64.0 MiB |  63.2 MiB/s |#####              | [38;2;255;136;0m 29%[39m | ETA:   0:00:00

.. parsed-literal::

     22.7 MiB of 64.0 MiB |  65.0 MiB/s |######             | [38;2;255;154;0m 35%[39m | ETA:   0:00:00

.. parsed-literal::

     26.7 MiB of 64.0 MiB |  65.0 MiB/s |#######            | [38;2;255;168;0m 41%[39m | ETA:   0:00:00

.. parsed-literal::

     30.8 MiB of 64.0 MiB |  66.0 MiB/s |#########          | [38;2;255;204;0m 48%[39m | ETA:   0:00:00

.. parsed-literal::

     34.8 MiB of 64.0 MiB |  66.0 MiB/s |##########         | [38;2;255;239;0m 54%[39m | ETA:   0:00:00

.. parsed-literal::

     38.9 MiB of 64.0 MiB |  66.4 MiB/s |###########        | [38;2;244;255;0m 60%[39m | ETA:   0:00:00

.. parsed-literal::

     42.9 MiB of 64.0 MiB |  66.4 MiB/s |############       | [38;2;227;255;0m 67%[39m | ETA:   0:00:00

.. parsed-literal::

     47.0 MiB of 64.0 MiB |  66.9 MiB/s |#############      | [38;2;209;255;0m 73%[39m | ETA:   0:00:00

.. parsed-literal::

     51.0 MiB of 64.0 MiB |  66.9 MiB/s |###############    | [38;2;191;255;0m 79%[39m | ETA:   0:00:00

.. parsed-literal::

     55.1 MiB of 64.0 MiB |  67.2 MiB/s |################   | [38;2;173;255;0m 86%[39m | ETA:   0:00:00

.. parsed-literal::

     59.1 MiB of 64.0 MiB |  67.2 MiB/s |#################  | [38;2;93;255;0m 92%[39m | ETA:   0:00:00

.. parsed-literal::

     63.2 MiB of 64.0 MiB |  67.5 MiB/s |################## | [38;2;15;255;0m 98%[39m | ETA:   0:00:00

.. parsed-literal::

    Downloading SFD data file to /home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_sgp.fits


.. parsed-literal::

      0.0 B of 64.0 MiB |   0.0 s/B |                       | [38;2;255;0;0m  0%[39m | ETA:  --:--:--

.. parsed-literal::

    Downloading data to '/home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_sgp.fits' ...
    Downloading https://dataverse.harvard.edu/api/access/datafile/2902695 ...


.. parsed-literal::

      2.4 MiB of 64.0 MiB |  41.4 MiB/s |                   | [38;2;255;25;0m  3%[39m | ETA:  00:00:00

.. parsed-literal::

      6.5 MiB of 64.0 MiB |  54.3 MiB/s |#                  | [38;2;255;83;0m 10%[39m | ETA:   0:00:01

.. parsed-literal::

     10.5 MiB of 64.0 MiB |  54.3 MiB/s |###                | [38;2;255;101;0m 16%[39m | ETA:   0:00:00

.. parsed-literal::

     14.6 MiB of 64.0 MiB |  59.8 MiB/s |####               | [38;2;255;118;0m 22%[39m | ETA:   0:00:00

.. parsed-literal::

     18.6 MiB of 64.0 MiB |  59.8 MiB/s |#####              | [38;2;255;136;0m 29%[39m | ETA:   0:00:00

.. parsed-literal::

     22.7 MiB of 64.0 MiB |  61.6 MiB/s |######             | [38;2;255;154;0m 35%[39m | ETA:   0:00:00

.. parsed-literal::

     26.7 MiB of 64.0 MiB |  61.6 MiB/s |#######            | [38;2;255;168;0m 41%[39m | ETA:   0:00:00

.. parsed-literal::

     30.8 MiB of 64.0 MiB |  62.9 MiB/s |#########          | [38;2;255;204;0m 48%[39m | ETA:   0:00:00

.. parsed-literal::

     34.8 MiB of 64.0 MiB |  62.9 MiB/s |##########         | [38;2;255;239;0m 54%[39m | ETA:   0:00:00

.. parsed-literal::

     38.9 MiB of 64.0 MiB |  63.7 MiB/s |###########        | [38;2;244;255;0m 60%[39m | ETA:   0:00:00

.. parsed-literal::

     42.9 MiB of 64.0 MiB |  63.7 MiB/s |############       | [38;2;227;255;0m 67%[39m | ETA:   0:00:00

.. parsed-literal::

     47.0 MiB of 64.0 MiB |  64.2 MiB/s |#############      | [38;2;209;255;0m 73%[39m | ETA:   0:00:00

.. parsed-literal::

     51.0 MiB of 64.0 MiB |  64.2 MiB/s |###############    | [38;2;191;255;0m 79%[39m | ETA:   0:00:00

.. parsed-literal::

     55.1 MiB of 64.0 MiB |  64.4 MiB/s |################   | [38;2;173;255;0m 86%[39m | ETA:   0:00:00

.. parsed-literal::

     59.1 MiB of 64.0 MiB |  64.4 MiB/s |#################  | [38;2;93;255;0m 92%[39m | ETA:   0:00:00

.. parsed-literal::

     63.2 MiB of 64.0 MiB |  64.7 MiB/s |################## | [38;2;15;255;0m 98%[39m | ETA:   0:00:00

.. code:: ipython3

    deredden_data = dereddener(mags_data)


.. parsed-literal::

    Inserting handle into data store.  output_dereddener: inprogress_output_dereddener.pq, dereddener


.. code:: ipython3

    deredden_data().keys()




.. parsed-literal::

    Index(['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst',
           'mag_y_lsst'],
          dtype='object')



We see that the deredden stage returns us a dictionary with the
dereddened magnitudes. Let’s plot the difference of the un-dereddened
magnitudes and the dereddened ones for u-band to see if they are,
indeed, slightly brighter:

.. code:: ipython3

    delta_u_mag = mags_data()['mag_u_lsst'] - deredden_data()['mag_u_lsst']
    plt.figure(figsize=(8,6))
    plt.scatter(mags_data()['mag_u_lsst'], delta_u_mag, s=15)
    plt.xlabel("orignal u-band mag", fontsize=12)
    plt.ylabel("u - deredden_u");



.. image:: ../../../docs/rendered/core_examples/FluxtoMag_and_Deredden_example_files/../../../docs/rendered/core_examples/FluxtoMag_and_Deredden_example_14_0.png


Clean up
~~~~~~~~

For cleanup, uncomment the line below to delete that SFD map directory
downloaded in this example:

.. code:: ipython3

    #! rm -rf sfd/
