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
    from rail.core.utils import find_rail_file
    from rail.tools.util_photometry import LSSTFluxToMagConverter, Dereddener

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
                      copy_cols=dict(ra='ra', decl='decl', objectId='objectId'))
    fluxToMag = LSSTFluxToMagConverter.make_stage(name='flux2mag', **ftomagdict)

.. code:: ipython3

    mags_data = fluxToMag(test_data)


.. parsed-literal::

    Inserting handle into data store.  output_flux2mag: inprogress_output_flux2mag.hdf5, flux2mag


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.13/x64/lib/python3.10/site-packages/rail/tools/util_photometry.py:379: RuntimeWarning: invalid value encountered in log10
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
     'decl',
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

     68.0 KiB of 64.0 MiB | 479.9 KiB/s |                   | [38;2;255;0;0m  0%[39m | ETA:   0:02:16

.. parsed-literal::

    289.0 KiB of 64.0 MiB | 1015.1 KiB/s |                  | [38;2;255;2;0m  0%[39m | ETA:   0:01:04

.. parsed-literal::

    830.0 KiB of 64.0 MiB | 1015.1 KiB/s |                  | [38;2;255;8;0m  1%[39m | ETA:   0:01:03

.. parsed-literal::

      1.1 MiB of 64.0 MiB |   2.2 MiB/s |                   | [38;2;255;11;0m  1%[39m | ETA:   0:00:28

.. parsed-literal::

      1.6 MiB of 64.0 MiB |   2.2 MiB/s |                   | [38;2;255;16;0m  2%[39m | ETA:   0:00:28

.. parsed-literal::

      1.9 MiB of 64.0 MiB |   2.6 MiB/s |                   | [38;2;255;19;0m  2%[39m | ETA:   0:00:23

.. parsed-literal::

      2.4 MiB of 64.0 MiB |   2.6 MiB/s |                   | [38;2;255;25;0m  3%[39m | ETA:   0:00:23

.. parsed-literal::

      2.7 MiB of 64.0 MiB |   2.9 MiB/s |                   | [38;2;255;28;0m  4%[39m | ETA:   0:00:21

.. parsed-literal::

      3.2 MiB of 64.0 MiB |   2.9 MiB/s |                   | [38;2;255;33;0m  5%[39m | ETA:   0:00:20

.. parsed-literal::

      3.6 MiB of 64.0 MiB |   3.1 MiB/s |#                  | [38;2;255;37;0m  5%[39m | ETA:   0:00:19

.. parsed-literal::

      4.1 MiB of 64.0 MiB |   3.1 MiB/s |#                  | [38;2;255;42;0m  6%[39m | ETA:   0:00:19

.. parsed-literal::

      4.5 MiB of 64.0 MiB |   3.3 MiB/s |#                  | [38;2;255;46;0m  7%[39m | ETA:   0:00:18

.. parsed-literal::

      4.9 MiB of 64.0 MiB |   3.3 MiB/s |#                  | [38;2;255;50;0m  7%[39m | ETA:   0:00:17

.. parsed-literal::

      5.4 MiB of 64.0 MiB |   3.4 MiB/s |#                  | [38;2;255;78;0m  8%[39m | ETA:   0:00:17

.. parsed-literal::

      6.1 MiB of 64.0 MiB |   3.5 MiB/s |#                  | [38;2;255;81;0m  9%[39m | ETA:   0:00:16

.. parsed-literal::

      6.5 MiB of 64.0 MiB |   3.5 MiB/s |#                  | [38;2;255;83;0m 10%[39m | ETA:   0:00:16

.. parsed-literal::

      7.0 MiB of 64.0 MiB |   3.6 MiB/s |##                 | [38;2;255;85;0m 10%[39m | ETA:   0:00:15

.. parsed-literal::

      7.4 MiB of 64.0 MiB |   3.6 MiB/s |##                 | [38;2;255;87;0m 11%[39m | ETA:   0:00:15

.. parsed-literal::

      8.0 MiB of 64.0 MiB |   3.7 MiB/s |##                 | [38;2;255;89;0m 12%[39m | ETA:   0:00:15

.. parsed-literal::

      8.6 MiB of 64.0 MiB |   3.8 MiB/s |##                 | [38;2;255;92;0m 13%[39m | ETA:   0:00:14

.. parsed-literal::

      9.0 MiB of 64.0 MiB |   3.8 MiB/s |##                 | [38;2;255;94;0m 14%[39m | ETA:   0:00:14

.. parsed-literal::

      9.6 MiB of 64.0 MiB |   3.8 MiB/s |##                 | [38;2;255;97;0m 15%[39m | ETA:   0:00:14

.. parsed-literal::

     10.0 MiB of 64.0 MiB |   3.8 MiB/s |##                 | [38;2;255;98;0m 15%[39m | ETA:   0:00:14

.. parsed-literal::

     10.5 MiB of 64.0 MiB |   3.8 MiB/s |###                | [38;2;255;101;0m 16%[39m | ETA:   0:00:13

.. parsed-literal::

     11.0 MiB of 64.0 MiB |   3.9 MiB/s |###                | [38;2;255;103;0m 17%[39m | ETA:   0:00:13

.. parsed-literal::

     11.7 MiB of 64.0 MiB |   4.0 MiB/s |###                | [38;2;255;106;0m 18%[39m | ETA:   0:00:12

.. parsed-literal::

     12.1 MiB of 64.0 MiB |   4.0 MiB/s |###                | [38;2;255;107;0m 18%[39m | ETA:   0:00:13

.. parsed-literal::

     12.7 MiB of 64.0 MiB |   4.1 MiB/s |###                | [38;2;255;110;0m 19%[39m | ETA:   0:00:12

.. parsed-literal::

     13.0 MiB of 64.0 MiB |   4.1 MiB/s |###                | [38;2;255;111;0m 20%[39m | ETA:   0:00:12

.. parsed-literal::

     13.5 MiB of 64.0 MiB |   4.1 MiB/s |###                | [38;2;255;113;0m 21%[39m | ETA:   0:00:12

.. parsed-literal::

     14.1 MiB of 64.0 MiB |   4.1 MiB/s |####               | [38;2;255;116;0m 22%[39m | ETA:   0:00:12

.. parsed-literal::

     14.5 MiB of 64.0 MiB |   4.1 MiB/s |####               | [38;2;255;118;0m 22%[39m | ETA:   0:00:11

.. parsed-literal::

     15.2 MiB of 64.0 MiB |   4.2 MiB/s |####               | [38;2;255;121;0m 23%[39m | ETA:   0:00:11

.. parsed-literal::

     15.4 MiB of 64.0 MiB |   4.2 MiB/s |####               | [38;2;255;122;0m 24%[39m | ETA:   0:00:11

.. parsed-literal::

     16.0 MiB of 64.0 MiB |   4.2 MiB/s |####               | [38;2;255;124;0m 24%[39m | ETA:   0:00:11

.. parsed-literal::

     16.5 MiB of 64.0 MiB |   4.2 MiB/s |####               | [38;2;255;127;0m 25%[39m | ETA:   0:00:11

.. parsed-literal::

     17.0 MiB of 64.0 MiB |   4.2 MiB/s |#####              | [38;2;255;129;0m 26%[39m | ETA:   0:00:11

.. parsed-literal::

     17.4 MiB of 64.0 MiB |   4.3 MiB/s |#####              | [38;2;255;131;0m 27%[39m | ETA:   0:00:10

.. parsed-literal::

     17.8 MiB of 64.0 MiB |   4.3 MiB/s |#####              | [38;2;255;132;0m 27%[39m | ETA:   0:00:10

.. parsed-literal::

     18.5 MiB of 64.0 MiB |   4.3 MiB/s |#####              | [38;2;255;135;0m 28%[39m | ETA:   0:00:10

.. parsed-literal::

     19.0 MiB of 64.0 MiB |   4.3 MiB/s |#####              | [38;2;255;138;0m 29%[39m | ETA:   0:00:10

.. parsed-literal::

     19.4 MiB of 64.0 MiB |   4.3 MiB/s |#####              | [38;2;255;140;0m 30%[39m | ETA:   0:00:10

.. parsed-literal::

     20.0 MiB of 64.0 MiB |   4.3 MiB/s |#####              | [38;2;255;142;0m 31%[39m | ETA:   0:00:10

.. parsed-literal::

     20.5 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;144;0m 32%[39m | ETA:   0:00:09

.. parsed-literal::

     21.1 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;147;0m 32%[39m | ETA:   0:00:09

.. parsed-literal::

     21.6 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;149;0m 33%[39m | ETA:   0:00:09

.. parsed-literal::

     21.9 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;150;0m 34%[39m | ETA:   0:00:09

.. parsed-literal::

     22.5 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;153;0m 35%[39m | ETA:   0:00:09

.. parsed-literal::

     23.1 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;155;0m 36%[39m | ETA:   0:00:09

.. parsed-literal::

     23.5 MiB of 64.0 MiB |   4.4 MiB/s |######             | [38;2;255;157;0m 36%[39m | ETA:   0:00:09

.. parsed-literal::

     24.0 MiB of 64.0 MiB |   4.5 MiB/s |#######            | [38;2;255;159;0m 37%[39m | ETA:   0:00:08

.. parsed-literal::

     24.6 MiB of 64.0 MiB |   4.5 MiB/s |#######            | [38;2;255;162;0m 38%[39m | ETA:   0:00:08

.. parsed-literal::

     25.1 MiB of 64.0 MiB |   4.5 MiB/s |#######            | [38;2;255;164;0m 39%[39m | ETA:   0:00:08

.. parsed-literal::

     25.6 MiB of 64.0 MiB |   4.5 MiB/s |#######            | [38;2;255;167;0m 40%[39m | ETA:   0:00:08

.. parsed-literal::

     25.9 MiB of 64.0 MiB |   4.5 MiB/s |#######            | [38;2;255;168;0m 40%[39m | ETA:   0:00:08

.. parsed-literal::

     26.6 MiB of 64.0 MiB |   4.5 MiB/s |#######            | [38;2;255;171;0m 41%[39m | ETA:   0:00:08

.. parsed-literal::

     27.1 MiB of 64.0 MiB |   4.5 MiB/s |########           | [38;2;255;172;0m 42%[39m | ETA:   0:00:08

.. parsed-literal::

     27.5 MiB of 64.0 MiB |   4.5 MiB/s |########           | [38;2;255;176;0m 43%[39m | ETA:   0:00:08

.. parsed-literal::

     28.0 MiB of 64.0 MiB |   4.6 MiB/s |########           | [38;2;255;180;0m 43%[39m | ETA:   0:00:07

.. parsed-literal::

     28.6 MiB of 64.0 MiB |   4.6 MiB/s |########           | [38;2;255;185;0m 44%[39m | ETA:   0:00:07

.. parsed-literal::

     29.1 MiB of 64.0 MiB |   4.6 MiB/s |########           | [38;2;255;189;0m 45%[39m | ETA:   0:00:07

.. parsed-literal::

     29.7 MiB of 64.0 MiB |   4.6 MiB/s |########           | [38;2;255;194;0m 46%[39m | ETA:   0:00:07

.. parsed-literal::

     30.0 MiB of 64.0 MiB |   4.6 MiB/s |########           | [38;2;255;197;0m 46%[39m | ETA:   0:00:07

.. parsed-literal::

     30.6 MiB of 64.0 MiB |   4.6 MiB/s |#########          | [38;2;255;202;0m 47%[39m | ETA:   0:00:07

.. parsed-literal::

     31.1 MiB of 64.0 MiB |   4.6 MiB/s |#########          | [38;2;255;206;0m 48%[39m | ETA:   0:00:07

.. parsed-literal::

     31.6 MiB of 64.0 MiB |   4.6 MiB/s |#########          | [38;2;255;211;0m 49%[39m | ETA:   0:00:07

.. parsed-literal::

     32.1 MiB of 64.0 MiB |   4.6 MiB/s |#########          | [38;2;255;215;0m 50%[39m | ETA:   0:00:06

.. parsed-literal::

     32.8 MiB of 64.0 MiB |   4.6 MiB/s |#########          | [38;2;255;222;0m 51%[39m | ETA:   0:00:06

.. parsed-literal::

     33.2 MiB of 64.0 MiB |   4.6 MiB/s |#########          | [38;2;255;225;0m 51%[39m | ETA:   0:00:06

.. parsed-literal::

     33.9 MiB of 64.0 MiB |   4.6 MiB/s |##########         | [38;2;255;231;0m 52%[39m | ETA:   0:00:06

.. parsed-literal::

     34.6 MiB of 64.0 MiB |   4.6 MiB/s |##########         | [38;2;255;238;0m 54%[39m | ETA:   0:00:06

.. parsed-literal::

     35.2 MiB of 64.0 MiB |   4.7 MiB/s |##########         | [38;2;255;243;0m 55%[39m | ETA:   0:00:06

.. parsed-literal::

     35.6 MiB of 64.0 MiB |   4.7 MiB/s |##########         | [38;2;255;246;0m 55%[39m | ETA:   0:00:06

.. parsed-literal::

     36.1 MiB of 64.0 MiB |   4.7 MiB/s |##########         | [38;2;255;251;0m 56%[39m | ETA:   0:00:05

.. parsed-literal::

     36.9 MiB of 64.0 MiB |   4.7 MiB/s |##########         | [38;2;255;257;0m 57%[39m | ETA:   0:00:05

.. parsed-literal::

     37.3 MiB of 64.0 MiB |   4.7 MiB/s |###########        | [38;2;255;261;0m 58%[39m | ETA:   0:00:05

.. parsed-literal::

     38.0 MiB of 64.0 MiB |   4.7 MiB/s |###########        | [38;2;248;255;0m 59%[39m | ETA:   0:00:05

.. parsed-literal::

     38.4 MiB of 64.0 MiB |   4.7 MiB/s |###########        | [38;2;246;255;0m 60%[39m | ETA:   0:00:05

.. parsed-literal::

     38.9 MiB of 64.0 MiB |   4.7 MiB/s |###########        | [38;2;244;255;0m 60%[39m | ETA:   0:00:05

.. parsed-literal::

     39.4 MiB of 64.0 MiB |   4.7 MiB/s |###########        | [38;2;242;255;0m 61%[39m | ETA:   0:00:05

.. parsed-literal::

     40.0 MiB of 64.0 MiB |   4.7 MiB/s |###########        | [38;2;240;255;0m 62%[39m | ETA:   0:00:05

.. parsed-literal::

     40.5 MiB of 64.0 MiB |   4.7 MiB/s |############       | [38;2;237;255;0m 63%[39m | ETA:   0:00:04

.. parsed-literal::

     40.9 MiB of 64.0 MiB |   4.7 MiB/s |############       | [38;2;235;255;0m 63%[39m | ETA:   0:00:04

.. parsed-literal::

     41.3 MiB of 64.0 MiB |   4.7 MiB/s |############       | [38;2;234;255;0m 64%[39m | ETA:   0:00:04

.. parsed-literal::

     42.1 MiB of 64.0 MiB |   4.7 MiB/s |############       | [38;2;231;255;0m 65%[39m | ETA:   0:00:04

.. parsed-literal::

     42.5 MiB of 64.0 MiB |   4.7 MiB/s |############       | [38;2;229;255;0m 66%[39m | ETA:   0:00:04

.. parsed-literal::

     42.9 MiB of 64.0 MiB |   4.7 MiB/s |############       | [38;2;227;255;0m 67%[39m | ETA:   0:00:04

.. parsed-literal::

     43.5 MiB of 64.0 MiB |   4.8 MiB/s |############       | [38;2;224;255;0m 68%[39m | ETA:   0:00:04

.. parsed-literal::

     44.0 MiB of 64.0 MiB |   4.8 MiB/s |#############      | [38;2;222;255;0m 68%[39m | ETA:   0:00:04

.. parsed-literal::

     44.6 MiB of 64.0 MiB |   4.8 MiB/s |#############      | [38;2;220;255;0m 69%[39m | ETA:   0:00:04

.. parsed-literal::

     44.8 MiB of 64.0 MiB |   4.7 MiB/s |#############      | [38;2;219;255;0m 69%[39m | ETA:   0:00:04

.. parsed-literal::

     45.4 MiB of 64.0 MiB |   4.7 MiB/s |#############      | [38;2;216;255;0m 70%[39m | ETA:   0:00:03

.. parsed-literal::

     45.9 MiB of 64.0 MiB |   4.8 MiB/s |#############      | [38;2;214;255;0m 71%[39m | ETA:   0:00:03

.. parsed-literal::

     46.2 MiB of 64.0 MiB |   4.8 MiB/s |#############      | [38;2;212;255;0m 72%[39m | ETA:   0:00:03

.. parsed-literal::

     46.6 MiB of 64.0 MiB |   4.7 MiB/s |#############      | [38;2;211;255;0m 72%[39m | ETA:   0:00:03

.. parsed-literal::

     47.0 MiB of 64.0 MiB |   4.7 MiB/s |#############      | [38;2;209;255;0m 73%[39m | ETA:   0:00:03

.. parsed-literal::

     47.3 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;207;255;0m 73%[39m | ETA:   0:00:03

.. parsed-literal::

     47.8 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;205;255;0m 74%[39m | ETA:   0:00:03

.. parsed-literal::

     48.2 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;204;255;0m 75%[39m | ETA:   0:00:03

.. parsed-literal::

     48.6 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;202;255;0m 75%[39m | ETA:   0:00:03

.. parsed-literal::

     49.1 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;200;255;0m 76%[39m | ETA:   0:00:03

.. parsed-literal::

     49.2 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;199;255;0m 76%[39m | ETA:   0:00:03

.. parsed-literal::

     49.9 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;196;255;0m 77%[39m | ETA:   0:00:03

.. parsed-literal::

     50.2 MiB of 64.0 MiB |   4.7 MiB/s |##############     | [38;2;195;255;0m 78%[39m | ETA:   0:00:02

.. parsed-literal::

     50.6 MiB of 64.0 MiB |   4.7 MiB/s |###############    | [38;2;193;255;0m 78%[39m | ETA:   0:00:02

.. parsed-literal::

     50.9 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;192;255;0m 79%[39m | ETA:   0:00:02

.. parsed-literal::

     51.2 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;190;255;0m 80%[39m | ETA:   0:00:02

.. parsed-literal::

     51.6 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;189;255;0m 80%[39m | ETA:   0:00:02

.. parsed-literal::

     51.9 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;188;255;0m 81%[39m | ETA:   0:00:02

.. parsed-literal::

     52.2 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;186;255;0m 81%[39m | ETA:   0:00:02

.. parsed-literal::

     52.5 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;185;255;0m 82%[39m | ETA:   0:00:02

.. parsed-literal::

     52.9 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;183;255;0m 82%[39m | ETA:   0:00:02

.. parsed-literal::

     53.3 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;181;255;0m 83%[39m | ETA:   0:00:02

.. parsed-literal::

     53.7 MiB of 64.0 MiB |   4.6 MiB/s |###############    | [38;2;180;255;0m 83%[39m | ETA:   0:00:02

.. parsed-literal::

     54.0 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;178;255;0m 84%[39m | ETA:   0:00:02

.. parsed-literal::

     54.3 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;177;255;0m 84%[39m | ETA:   0:00:02

.. parsed-literal::

     54.7 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;175;255;0m 85%[39m | ETA:   0:00:02

.. parsed-literal::

     55.0 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;174;255;0m 85%[39m | ETA:   0:00:01

.. parsed-literal::

     55.4 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;172;255;0m 86%[39m | ETA:   0:00:01

.. parsed-literal::

     55.8 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;171;255;0m 87%[39m | ETA:   0:00:01

.. parsed-literal::

     56.1 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;169;255;0m 87%[39m | ETA:   0:00:01

.. parsed-literal::

     56.5 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;167;255;0m 88%[39m | ETA:   0:00:01

.. parsed-literal::

     56.7 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;166;255;0m 88%[39m | ETA:   0:00:01

.. parsed-literal::

     57.1 MiB of 64.0 MiB |   4.5 MiB/s |################   | [38;2;165;255;0m 89%[39m | ETA:   0:00:01

.. parsed-literal::

     57.4 MiB of 64.0 MiB |   4.5 MiB/s |#################  | [38;2;163;255;0m 89%[39m | ETA:   0:00:01

.. parsed-literal::

     57.8 MiB of 64.0 MiB |   4.5 MiB/s |#################  | [38;2;161;255;0m 90%[39m | ETA:   0:00:01

.. parsed-literal::

     58.2 MiB of 64.0 MiB |   4.5 MiB/s |#################  | [38;2;160;255;0m 90%[39m | ETA:   0:00:01

.. parsed-literal::

     58.6 MiB of 64.0 MiB |   4.5 MiB/s |#################  | [38;2;158;255;0m 91%[39m | ETA:   0:00:01

.. parsed-literal::

     59.0 MiB of 64.0 MiB |   4.5 MiB/s |#################  | [38;2;96;255;0m 92%[39m | ETA:   0:00:01

.. parsed-literal::

     59.4 MiB of 64.0 MiB |   4.5 MiB/s |#################  | [38;2;88;255;0m 92%[39m | ETA:   0:00:01

.. parsed-literal::

     59.8 MiB of 64.0 MiB |   4.4 MiB/s |#################  | [38;2;80;255;0m 93%[39m | ETA:   0:00:00

.. parsed-literal::

     60.2 MiB of 64.0 MiB |   4.4 MiB/s |#################  | [38;2;72;255;0m 94%[39m | ETA:   0:00:00

.. parsed-literal::

     60.6 MiB of 64.0 MiB |   4.4 MiB/s |#################  | [38;2;65;255;0m 94%[39m | ETA:   0:00:00

.. parsed-literal::

     61.0 MiB of 64.0 MiB |   4.4 MiB/s |################## | [38;2;57;255;0m 95%[39m | ETA:   0:00:00

.. parsed-literal::

     61.5 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;48;255;0m 96%[39m | ETA:   0:00:00

.. parsed-literal::

     61.8 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;43;255;0m 96%[39m | ETA:   0:00:00

.. parsed-literal::

     62.0 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;37;255;0m 96%[39m | ETA:   0:00:00

.. parsed-literal::

     62.3 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;32;255;0m 97%[39m | ETA:   0:00:00

.. parsed-literal::

     62.6 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;26;255;0m 97%[39m | ETA:   0:00:00

.. parsed-literal::

     62.9 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;20;255;0m 98%[39m | ETA:   0:00:00

.. parsed-literal::

     63.2 MiB of 64.0 MiB |   4.5 MiB/s |################## | [38;2;15;255;0m 98%[39m | ETA:   0:00:00

.. parsed-literal::

     63.5 MiB of 64.0 MiB |   4.4 MiB/s |################## | [38;2;9;255;0m 99%[39m | ETA:   0:00:00

.. parsed-literal::

     63.8 MiB of 64.0 MiB |   4.4 MiB/s |################## | [38;2;3;255;0m 99%[39m | ETA:   0:00:00

.. parsed-literal::

     64.0 MiB of 64.0 MiB |   4.4 MiB/s |###################| [38;2;0;255;0m100%[39m | ETA:  00:00:00

.. parsed-literal::

    Downloading SFD data file to /home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_sgp.fits


.. parsed-literal::

    Downloading data to '/home/runner/work/rail_notebooks/rail_notebooks/rail/examples/core_examples/sfd/SFD_dust_4096_sgp.fits' ...
    Downloading https://dataverse.harvard.edu/api/access/datafile/2902695 ...


.. parsed-literal::

      0.0 B of 64.0 MiB |   0.0 s/B |                       | [38;2;255;0;0m  0%[39m | ETA:  --:--:--

.. parsed-literal::

     67.0 KiB of 64.0 MiB | 502.3 KiB/s |                   | [38;2;255;0;0m  0%[39m | ETA:   0:02:10

.. parsed-literal::

    288.0 KiB of 64.0 MiB |   1.1 MiB/s |                   | [38;2;255;2;0m  0%[39m | ETA:   0:01:00

.. parsed-literal::

    830.0 KiB of 64.0 MiB |   1.1 MiB/s |                   | [38;2;255;8;0m  1%[39m | ETA:   0:01:00

.. parsed-literal::

      1.6 MiB of 64.0 MiB |   4.0 MiB/s |                   | [38;2;255;16;0m  2%[39m | ETA:   0:00:15

.. parsed-literal::

      2.4 MiB of 64.0 MiB |   4.0 MiB/s |                   | [38;2;255;25;0m  3%[39m | ETA:   0:00:15

.. parsed-literal::

      4.9 MiB of 64.0 MiB |   9.0 MiB/s |#                  | [38;2;255;50;0m  7%[39m | ETA:   0:00:06

.. parsed-literal::

      6.5 MiB of 64.0 MiB |   9.0 MiB/s |#                  | [38;2;255;83;0m 10%[39m | ETA:   0:00:06

.. parsed-literal::

      8.9 MiB of 64.0 MiB |  13.1 MiB/s |##                 | [38;2;255;93;0m 13%[39m | ETA:   0:00:04

.. parsed-literal::

     10.5 MiB of 64.0 MiB |  13.1 MiB/s |###                | [38;2;255;101;0m 16%[39m | ETA:   0:00:04

.. parsed-literal::

     13.0 MiB of 64.0 MiB |  15.9 MiB/s |###                | [38;2;255;111;0m 20%[39m | ETA:   0:00:03

.. parsed-literal::

     14.6 MiB of 64.0 MiB |  15.9 MiB/s |####               | [38;2;255;118;0m 22%[39m | ETA:   0:00:03

.. parsed-literal::

     17.0 MiB of 64.0 MiB |  17.9 MiB/s |#####              | [38;2;255;129;0m 26%[39m | ETA:   0:00:02

.. parsed-literal::

     18.6 MiB of 64.0 MiB |  17.9 MiB/s |#####              | [38;2;255;136;0m 29%[39m | ETA:   0:00:02

.. parsed-literal::

     20.3 MiB of 64.0 MiB |  18.8 MiB/s |######             | [38;2;255;143;0m 31%[39m | ETA:   0:00:02

.. parsed-literal::

     22.7 MiB of 64.0 MiB |  18.8 MiB/s |######             | [38;2;255;154;0m 35%[39m | ETA:   0:00:02

.. parsed-literal::

     24.3 MiB of 64.0 MiB |  20.1 MiB/s |#######            | [38;2;255;161;0m 37%[39m | ETA:   0:00:01

.. parsed-literal::

     26.7 MiB of 64.0 MiB |  20.1 MiB/s |#######            | [38;2;255;168;0m 41%[39m | ETA:   0:00:01

.. parsed-literal::

     28.4 MiB of 64.0 MiB |  21.0 MiB/s |########           | [38;2;255;183;0m 44%[39m | ETA:   0:00:01

.. parsed-literal::

     30.8 MiB of 64.0 MiB |  21.0 MiB/s |#########          | [38;2;255;204;0m 48%[39m | ETA:   0:00:01

.. parsed-literal::

     32.4 MiB of 64.0 MiB |  21.9 MiB/s |#########          | [38;2;255;218;0m 50%[39m | ETA:   0:00:01

.. parsed-literal::

     34.8 MiB of 64.0 MiB |  21.9 MiB/s |##########         | [38;2;255;239;0m 54%[39m | ETA:   0:00:01

.. parsed-literal::

     36.5 MiB of 64.0 MiB |  22.5 MiB/s |##########         | [38;2;255;253;0m 56%[39m | ETA:   0:00:01

.. parsed-literal::

     38.9 MiB of 64.0 MiB |  22.5 MiB/s |###########        | [38;2;244;255;0m 60%[39m | ETA:   0:00:01

.. parsed-literal::

     40.5 MiB of 64.0 MiB |  23.1 MiB/s |############       | [38;2;237;255;0m 63%[39m | ETA:   0:00:01

.. parsed-literal::

     42.9 MiB of 64.0 MiB |  23.1 MiB/s |############       | [38;2;227;255;0m 67%[39m | ETA:   0:00:00

.. parsed-literal::

     44.6 MiB of 64.0 MiB |  23.6 MiB/s |#############      | [38;2;220;255;0m 69%[39m | ETA:   0:00:00

.. parsed-literal::

     46.6 MiB of 64.0 MiB |  23.0 MiB/s |#############      | [38;2;210;255;0m 72%[39m | ETA:   0:00:00

.. parsed-literal::

     50.2 MiB of 64.0 MiB |  23.0 MiB/s |##############     | [38;2;195;255;0m 78%[39m | ETA:   0:00:00

.. parsed-literal::

     51.9 MiB of 64.0 MiB |  24.0 MiB/s |###############    | [38;2;188;255;0m 81%[39m | ETA:   0:00:00

.. parsed-literal::

     54.3 MiB of 64.0 MiB |  24.0 MiB/s |################   | [38;2;177;255;0m 84%[39m | ETA:   0:00:00

.. parsed-literal::

     56.7 MiB of 64.0 MiB |  24.6 MiB/s |################   | [38;2;166;255;0m 88%[39m | ETA:   0:00:00

.. parsed-literal::

     59.1 MiB of 64.0 MiB |  24.6 MiB/s |#################  | [38;2;93;255;0m 92%[39m | ETA:   0:00:00

.. parsed-literal::

     61.6 MiB of 64.0 MiB |  25.3 MiB/s |################## | [38;2;46;255;0m 96%[39m | ETA:   0:00:00

.. parsed-literal::

     63.2 MiB of 64.0 MiB |  25.3 MiB/s |################## | [38;2;15;255;0m 98%[39m | ETA:   0:00:00

.. code:: ipython3

    deredden_data = dereddener(mags_data)


.. parsed-literal::

    Inserting handle into data store.  output_dereddener: inprogress_output_dereddener.hdf5, dereddener


.. code:: ipython3

    deredden_data().keys()




.. parsed-literal::

    dict_keys(['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst'])



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
