GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4367653e-01	 3.2058929e-01	-3.3396927e-01	 3.2057952e-01	[-3.3452015e-01]	 4.5902491e-01


.. parsed-literal::

       2	-2.7151176e-01	 3.0945398e-01	-2.4726752e-01	 3.0939615e-01	[-2.4739719e-01]	 2.2989416e-01


.. parsed-literal::

       3	-2.2811421e-01	 2.8997081e-01	-1.8739556e-01	 2.8633380e-01	[-1.7507136e-01]	 2.7880120e-01
       4	-1.9956264e-01	 2.6637255e-01	-1.6071094e-01	 2.6488751e-01	[-1.3890003e-01]	 1.7115545e-01


.. parsed-literal::

       5	-1.0814952e-01	 2.5780004e-01	-7.2039565e-02	 2.5378246e-01	[-5.2813370e-02]	 2.1248889e-01


.. parsed-literal::

       6	-7.3983779e-02	 2.5225389e-01	-4.2050486e-02	 2.5341661e-01	[-3.6406605e-02]	 2.1408963e-01


.. parsed-literal::

       7	-5.4921175e-02	 2.4928474e-01	-3.0413415e-02	 2.4712321e-01	[-1.8402962e-02]	 2.1627688e-01


.. parsed-literal::

       8	-4.3448088e-02	 2.4741075e-01	-2.2611516e-02	 2.4458708e-01	[-9.0620004e-03]	 2.1403360e-01


.. parsed-literal::

       9	-2.8563903e-02	 2.4465885e-01	-1.0950086e-02	 2.4207345e-01	[ 2.0179119e-03]	 2.0657277e-01


.. parsed-literal::

      10	-2.0066164e-02	 2.4346300e-01	-4.8355380e-03	 2.4122490e-01	[ 7.3605888e-03]	 2.1822000e-01


.. parsed-literal::

      11	-1.3022929e-02	 2.4190092e-01	 1.4371250e-03	 2.4007410e-01	[ 1.2735180e-02]	 2.1692038e-01
      12	-1.0868984e-02	 2.4148932e-01	 3.4129788e-03	 2.3963564e-01	[ 1.4505026e-02]	 1.8182373e-01


.. parsed-literal::

      13	-4.1016438e-03	 2.4027958e-01	 9.9676443e-03	 2.3874283e-01	[ 1.9148364e-02]	 2.0725036e-01


.. parsed-literal::

      14	 3.8805548e-02	 2.2552804e-01	 5.9368749e-02	 2.3050944e-01	[ 5.6004358e-02]	 4.1578484e-01


.. parsed-literal::

      15	 2.0856645e-01	 2.1952607e-01	 2.3436720e-01	 2.2587036e-01	[ 2.3130172e-01]	 2.0632291e-01


.. parsed-literal::

      16	 2.4206146e-01	 2.1523835e-01	 2.7587422e-01	 2.1895599e-01	[ 2.8455412e-01]	 2.1630764e-01


.. parsed-literal::

      17	 2.9344513e-01	 2.1313159e-01	 3.2630082e-01	 2.2019986e-01	[ 3.2501568e-01]	 2.0990992e-01


.. parsed-literal::

      18	 3.3307343e-01	 2.0841942e-01	 3.6543689e-01	 2.1395612e-01	[ 3.6121281e-01]	 2.1602011e-01


.. parsed-literal::

      19	 3.7557107e-01	 2.0335129e-01	 4.0736636e-01	 2.0918324e-01	[ 3.9843160e-01]	 2.1086621e-01


.. parsed-literal::

      20	 4.7011670e-01	 1.9738995e-01	 5.0245129e-01	 2.0564273e-01	[ 4.7986298e-01]	 2.2021461e-01
      21	 5.6890469e-01	 1.9710537e-01	 6.0486491e-01	 2.0396651e-01	[ 5.7253644e-01]	 2.0959759e-01


.. parsed-literal::

      22	 6.3744840e-01	 1.9952947e-01	 6.7485758e-01	 2.0451633e-01	[ 6.3432434e-01]	 2.0849133e-01


.. parsed-literal::

      23	 6.7658594e-01	 1.9376812e-01	 7.1459749e-01	 1.9849730e-01	[ 6.7177639e-01]	 2.1738219e-01


.. parsed-literal::

      24	 7.0452791e-01	 1.9157700e-01	 7.4206784e-01	 1.9548411e-01	[ 7.0184607e-01]	 2.0618606e-01
      25	 7.3431828e-01	 1.9530986e-01	 7.7206349e-01	 1.9577939e-01	[ 7.2853453e-01]	 1.8651438e-01


.. parsed-literal::

      26	 7.5602491e-01	 1.9510149e-01	 7.9333359e-01	 1.9400526e-01	[ 7.5620379e-01]	 2.0977759e-01
      27	 7.7577773e-01	 1.9259388e-01	 8.1330674e-01	 1.9269838e-01	[ 7.7622509e-01]	 1.8556547e-01


.. parsed-literal::

      28	 7.9969165e-01	 1.9148149e-01	 8.3866143e-01	 1.9091991e-01	[ 8.0215814e-01]	 1.9998741e-01
      29	 8.4104894e-01	 1.8567155e-01	 8.8098057e-01	 1.8532184e-01	[ 8.4736489e-01]	 1.9485331e-01


.. parsed-literal::

      30	 8.6287626e-01	 1.8760814e-01	 9.0328210e-01	 1.8964381e-01	[ 8.7197157e-01]	 1.9840193e-01


.. parsed-literal::

      31	 8.9363449e-01	 1.8603288e-01	 9.3494469e-01	 1.8755308e-01	[ 9.0212187e-01]	 2.1024871e-01
      32	 9.1251945e-01	 1.8452457e-01	 9.5345757e-01	 1.8686472e-01	[ 9.2399986e-01]	 1.8392587e-01


.. parsed-literal::

      33	 9.3006389e-01	 1.8180389e-01	 9.7172375e-01	 1.8287964e-01	[ 9.4689790e-01]	 1.8501687e-01
      34	 9.4561094e-01	 1.8084246e-01	 9.8774965e-01	 1.8156616e-01	[ 9.6224849e-01]	 1.7702985e-01


.. parsed-literal::

      35	 9.6404588e-01	 1.7980003e-01	 1.0069023e+00	 1.8091633e-01	[ 9.7951907e-01]	 2.0706272e-01
      36	 9.8331177e-01	 1.7749106e-01	 1.0281426e+00	 1.8062151e-01	[ 9.9253385e-01]	 1.7301798e-01


.. parsed-literal::

      37	 1.0027967e+00	 1.7634318e-01	 1.0476447e+00	 1.7926016e-01	[ 1.0015657e+00]	 1.9196224e-01


.. parsed-literal::

      38	 1.0116858e+00	 1.7579031e-01	 1.0561080e+00	 1.7833182e-01	[ 1.0062660e+00]	 2.0266294e-01


.. parsed-literal::

      39	 1.0244561e+00	 1.7339686e-01	 1.0689845e+00	 1.7680539e-01	[ 1.0127485e+00]	 2.1107745e-01
      40	 1.0358919e+00	 1.7026058e-01	 1.0808592e+00	 1.7205680e-01	[ 1.0242362e+00]	 1.9145060e-01


.. parsed-literal::

      41	 1.0461507e+00	 1.6819994e-01	 1.0915548e+00	 1.6971408e-01	[ 1.0338009e+00]	 1.8661499e-01


.. parsed-literal::

      42	 1.0579347e+00	 1.6634948e-01	 1.1037633e+00	 1.6769752e-01	[ 1.0461672e+00]	 2.0549464e-01
      43	 1.0757075e+00	 1.6393397e-01	 1.1219563e+00	 1.6531143e-01	[ 1.0601239e+00]	 1.7858815e-01


.. parsed-literal::

      44	 1.0870467e+00	 1.6191921e-01	 1.1341274e+00	 1.6280716e-01	[ 1.0718737e+00]	 2.0357537e-01


.. parsed-literal::

      45	 1.1027617e+00	 1.5968092e-01	 1.1493721e+00	 1.6105282e-01	[ 1.0841350e+00]	 2.0999861e-01


.. parsed-literal::

      46	 1.1111715e+00	 1.5893987e-01	 1.1574056e+00	 1.6002980e-01	[ 1.0906590e+00]	 2.0728683e-01


.. parsed-literal::

      47	 1.1204566e+00	 1.5687156e-01	 1.1673787e+00	 1.5854527e-01	[ 1.0959511e+00]	 2.0574760e-01
      48	 1.1304755e+00	 1.5555683e-01	 1.1777969e+00	 1.5821759e-01	[ 1.1036424e+00]	 1.8849111e-01


.. parsed-literal::

      49	 1.1408043e+00	 1.5366878e-01	 1.1884178e+00	 1.5718458e-01	[ 1.1137477e+00]	 2.1214771e-01
      50	 1.1519980e+00	 1.5158502e-01	 1.2000611e+00	 1.5548865e-01	[ 1.1232211e+00]	 1.7336679e-01


.. parsed-literal::

      51	 1.1614978e+00	 1.4954404e-01	 1.2099396e+00	 1.5300856e-01	[ 1.1323205e+00]	 2.0969820e-01
      52	 1.1710824e+00	 1.4849020e-01	 1.2195658e+00	 1.5187616e-01	[ 1.1393433e+00]	 1.9901061e-01


.. parsed-literal::

      53	 1.1828132e+00	 1.4698659e-01	 1.2314796e+00	 1.5038790e-01	[ 1.1466429e+00]	 2.0972776e-01


.. parsed-literal::

      54	 1.1930461e+00	 1.4607403e-01	 1.2421840e+00	 1.4986347e-01	[ 1.1529869e+00]	 2.1295047e-01


.. parsed-literal::

      55	 1.2036358e+00	 1.4510673e-01	 1.2525658e+00	 1.4931644e-01	[ 1.1542936e+00]	 2.0507097e-01


.. parsed-literal::

      56	 1.2114815e+00	 1.4446996e-01	 1.2605541e+00	 1.4954520e-01	[ 1.1549634e+00]	 2.1015286e-01


.. parsed-literal::

      57	 1.2200799e+00	 1.4375430e-01	 1.2693311e+00	 1.4967225e-01	  1.1459120e+00 	 2.0649719e-01


.. parsed-literal::

      58	 1.2299123e+00	 1.4254679e-01	 1.2793922e+00	 1.4854451e-01	  1.1514244e+00 	 2.0101810e-01


.. parsed-literal::

      59	 1.2405063e+00	 1.4157234e-01	 1.2903505e+00	 1.4751118e-01	[ 1.1594546e+00]	 2.0256329e-01


.. parsed-literal::

      60	 1.2495135e+00	 1.4101357e-01	 1.2999020e+00	 1.4607928e-01	[ 1.1608780e+00]	 2.1713519e-01


.. parsed-literal::

      61	 1.2573090e+00	 1.4074824e-01	 1.3077342e+00	 1.4578955e-01	[ 1.1714825e+00]	 2.1298695e-01


.. parsed-literal::

      62	 1.2659452e+00	 1.4051072e-01	 1.3164821e+00	 1.4588995e-01	[ 1.1744505e+00]	 2.1806335e-01
      63	 1.2769790e+00	 1.3996192e-01	 1.3280775e+00	 1.4623827e-01	  1.1594862e+00 	 1.7809129e-01


.. parsed-literal::

      64	 1.2827365e+00	 1.3989058e-01	 1.3337410e+00	 1.4670067e-01	  1.1433290e+00 	 1.9130659e-01


.. parsed-literal::

      65	 1.2892371e+00	 1.3931481e-01	 1.3399898e+00	 1.4621889e-01	  1.1550579e+00 	 2.1634579e-01


.. parsed-literal::

      66	 1.2956030e+00	 1.3858179e-01	 1.3464794e+00	 1.4543125e-01	  1.1584546e+00 	 2.1503520e-01
      67	 1.3018685e+00	 1.3823174e-01	 1.3528118e+00	 1.4533782e-01	  1.1668625e+00 	 1.7572355e-01


.. parsed-literal::

      68	 1.3077589e+00	 1.3783740e-01	 1.3588652e+00	 1.4556442e-01	  1.1675474e+00 	 1.8751693e-01
      69	 1.3141494e+00	 1.3775893e-01	 1.3651465e+00	 1.4572720e-01	[ 1.1752250e+00]	 1.8423200e-01


.. parsed-literal::

      70	 1.3202565e+00	 1.3771907e-01	 1.3711896e+00	 1.4591919e-01	[ 1.1801129e+00]	 2.0088315e-01
      71	 1.3282843e+00	 1.3732312e-01	 1.3793740e+00	 1.4618149e-01	  1.1772345e+00 	 1.9779992e-01


.. parsed-literal::

      72	 1.3308807e+00	 1.3728707e-01	 1.3824469e+00	 1.4583428e-01	  1.1730551e+00 	 2.0824838e-01
      73	 1.3408254e+00	 1.3662943e-01	 1.3921611e+00	 1.4573300e-01	  1.1741261e+00 	 1.9974637e-01


.. parsed-literal::

      74	 1.3449449e+00	 1.3627984e-01	 1.3963721e+00	 1.4550179e-01	  1.1701194e+00 	 2.0424104e-01


.. parsed-literal::

      75	 1.3509663e+00	 1.3584929e-01	 1.4027265e+00	 1.4520258e-01	  1.1612117e+00 	 2.0365357e-01
      76	 1.3587553e+00	 1.3539736e-01	 1.4109893e+00	 1.4487711e-01	  1.1446109e+00 	 1.7458797e-01


.. parsed-literal::

      77	 1.3639542e+00	 1.3513347e-01	 1.4165641e+00	 1.4544542e-01	  1.1459017e+00 	 3.1739306e-01


.. parsed-literal::

      78	 1.3688332e+00	 1.3500801e-01	 1.4215099e+00	 1.4565461e-01	  1.1472227e+00 	 2.0558453e-01


.. parsed-literal::

      79	 1.3751430e+00	 1.3448020e-01	 1.4278656e+00	 1.4558695e-01	  1.1564947e+00 	 2.1052122e-01
      80	 1.3808724e+00	 1.3401454e-01	 1.4337230e+00	 1.4591097e-01	  1.1544650e+00 	 1.9111609e-01


.. parsed-literal::

      81	 1.3867735e+00	 1.3362660e-01	 1.4395235e+00	 1.4550031e-01	  1.1684410e+00 	 2.0537877e-01


.. parsed-literal::

      82	 1.3915059e+00	 1.3324199e-01	 1.4442008e+00	 1.4475356e-01	  1.1773323e+00 	 2.1951246e-01


.. parsed-literal::

      83	 1.3964204e+00	 1.3297873e-01	 1.4491154e+00	 1.4443386e-01	[ 1.1820991e+00]	 2.1671557e-01
      84	 1.4013580e+00	 1.3256638e-01	 1.4539800e+00	 1.4303459e-01	[ 1.1902035e+00]	 1.8836570e-01


.. parsed-literal::

      85	 1.4060929e+00	 1.3242741e-01	 1.4586684e+00	 1.4331656e-01	[ 1.1925176e+00]	 2.0996165e-01


.. parsed-literal::

      86	 1.4115453e+00	 1.3218706e-01	 1.4642017e+00	 1.4348573e-01	[ 1.1935350e+00]	 2.2174120e-01


.. parsed-literal::

      87	 1.4163745e+00	 1.3185442e-01	 1.4690883e+00	 1.4312288e-01	[ 1.2003076e+00]	 2.2245240e-01


.. parsed-literal::

      88	 1.4210608e+00	 1.3135407e-01	 1.4739462e+00	 1.4210680e-01	[ 1.2015895e+00]	 3.3320498e-01


.. parsed-literal::

      89	 1.4258850e+00	 1.3082650e-01	 1.4788840e+00	 1.4149582e-01	[ 1.2162835e+00]	 2.0098376e-01
      90	 1.4282179e+00	 1.3066957e-01	 1.4811665e+00	 1.4090266e-01	[ 1.2192432e+00]	 1.8771195e-01


.. parsed-literal::

      91	 1.4312715e+00	 1.3035782e-01	 1.4842861e+00	 1.4028854e-01	[ 1.2216702e+00]	 2.0752048e-01


.. parsed-literal::

      92	 1.4359890e+00	 1.2979307e-01	 1.4892904e+00	 1.3963020e-01	  1.2190951e+00 	 2.1342802e-01
      93	 1.4404308e+00	 1.2955781e-01	 1.4938683e+00	 1.3941925e-01	  1.2202887e+00 	 1.9082141e-01


.. parsed-literal::

      94	 1.4438755e+00	 1.2943352e-01	 1.4972753e+00	 1.3998871e-01	[ 1.2259940e+00]	 2.1551085e-01


.. parsed-literal::

      95	 1.4480801e+00	 1.2940345e-01	 1.5014306e+00	 1.4067741e-01	[ 1.2317369e+00]	 2.0105100e-01


.. parsed-literal::

      96	 1.4523050e+00	 1.2927364e-01	 1.5057189e+00	 1.4138923e-01	[ 1.2468190e+00]	 2.0729542e-01


.. parsed-literal::

      97	 1.4570507e+00	 1.2912186e-01	 1.5105125e+00	 1.4143574e-01	[ 1.2581986e+00]	 2.1271706e-01
      98	 1.4606192e+00	 1.2886095e-01	 1.5140897e+00	 1.4065744e-01	[ 1.2651975e+00]	 2.0173192e-01


.. parsed-literal::

      99	 1.4637403e+00	 1.2867151e-01	 1.5172928e+00	 1.3994460e-01	[ 1.2673057e+00]	 2.1636367e-01


.. parsed-literal::

     100	 1.4673498e+00	 1.2861312e-01	 1.5210993e+00	 1.3946671e-01	  1.2612743e+00 	 2.1047568e-01


.. parsed-literal::

     101	 1.4691975e+00	 1.2858617e-01	 1.5230985e+00	 1.3917722e-01	  1.2554561e+00 	 2.0891452e-01


.. parsed-literal::

     102	 1.4712774e+00	 1.2858918e-01	 1.5250688e+00	 1.3926450e-01	  1.2542469e+00 	 2.1241474e-01


.. parsed-literal::

     103	 1.4729028e+00	 1.2858259e-01	 1.5266921e+00	 1.3948300e-01	  1.2534073e+00 	 2.0649409e-01


.. parsed-literal::

     104	 1.4753423e+00	 1.2854417e-01	 1.5291535e+00	 1.3948866e-01	  1.2501823e+00 	 2.0779777e-01


.. parsed-literal::

     105	 1.4776605e+00	 1.2852952e-01	 1.5315610e+00	 1.3968784e-01	  1.2436015e+00 	 2.1789861e-01
     106	 1.4802262e+00	 1.2840408e-01	 1.5341631e+00	 1.3932876e-01	  1.2411924e+00 	 1.7227769e-01


.. parsed-literal::

     107	 1.4822133e+00	 1.2827071e-01	 1.5361740e+00	 1.3909238e-01	  1.2398923e+00 	 2.1000695e-01


.. parsed-literal::

     108	 1.4849711e+00	 1.2818832e-01	 1.5390298e+00	 1.3878334e-01	  1.2335646e+00 	 2.1187639e-01


.. parsed-literal::

     109	 1.4875961e+00	 1.2839475e-01	 1.5418006e+00	 1.3862170e-01	  1.2212639e+00 	 2.1101356e-01


.. parsed-literal::

     110	 1.4903751e+00	 1.2866073e-01	 1.5446356e+00	 1.3860532e-01	  1.2152085e+00 	 2.1841335e-01


.. parsed-literal::

     111	 1.4920225e+00	 1.2875822e-01	 1.5462190e+00	 1.3853095e-01	  1.2146281e+00 	 2.1993113e-01


.. parsed-literal::

     112	 1.4938792e+00	 1.2878462e-01	 1.5480059e+00	 1.3857945e-01	  1.2171391e+00 	 2.1364880e-01
     113	 1.4959627e+00	 1.2873573e-01	 1.5500547e+00	 1.3850324e-01	  1.2160146e+00 	 1.9515419e-01


.. parsed-literal::

     114	 1.4981643e+00	 1.2859935e-01	 1.5522094e+00	 1.3851149e-01	  1.2184537e+00 	 2.0707011e-01
     115	 1.5000430e+00	 1.2843854e-01	 1.5540932e+00	 1.3876587e-01	  1.2231499e+00 	 1.9199729e-01


.. parsed-literal::

     116	 1.5014622e+00	 1.2839014e-01	 1.5555480e+00	 1.3866030e-01	  1.2185541e+00 	 2.0727539e-01


.. parsed-literal::

     117	 1.5030135e+00	 1.2829551e-01	 1.5571938e+00	 1.3858104e-01	  1.2114052e+00 	 2.1836734e-01
     118	 1.5050299e+00	 1.2817830e-01	 1.5594389e+00	 1.3858009e-01	  1.1929502e+00 	 2.0223808e-01


.. parsed-literal::

     119	 1.5069802e+00	 1.2803468e-01	 1.5614722e+00	 1.3845374e-01	  1.1826020e+00 	 2.1699882e-01


.. parsed-literal::

     120	 1.5083868e+00	 1.2791531e-01	 1.5628778e+00	 1.3838240e-01	  1.1765234e+00 	 2.4095225e-01


.. parsed-literal::

     121	 1.5103390e+00	 1.2774448e-01	 1.5648610e+00	 1.3824549e-01	  1.1651771e+00 	 2.2146177e-01


.. parsed-literal::

     122	 1.5117377e+00	 1.2762081e-01	 1.5663289e+00	 1.3801792e-01	  1.1472518e+00 	 3.1923342e-01
     123	 1.5134730e+00	 1.2750799e-01	 1.5680811e+00	 1.3775746e-01	  1.1377748e+00 	 1.8783402e-01


.. parsed-literal::

     124	 1.5152656e+00	 1.2747688e-01	 1.5699103e+00	 1.3742731e-01	  1.1294430e+00 	 1.8326640e-01


.. parsed-literal::

     125	 1.5172072e+00	 1.2755049e-01	 1.5719199e+00	 1.3707272e-01	  1.1208605e+00 	 2.0260334e-01
     126	 1.5188713e+00	 1.2784845e-01	 1.5737706e+00	 1.3666125e-01	  1.1112218e+00 	 1.9166446e-01


.. parsed-literal::

     127	 1.5210389e+00	 1.2784407e-01	 1.5759244e+00	 1.3656402e-01	  1.1126368e+00 	 2.1078992e-01


.. parsed-literal::

     128	 1.5221880e+00	 1.2770604e-01	 1.5770420e+00	 1.3639203e-01	  1.1158226e+00 	 2.1482158e-01


.. parsed-literal::

     129	 1.5233773e+00	 1.2758753e-01	 1.5781995e+00	 1.3640145e-01	  1.1260034e+00 	 2.0620441e-01
     130	 1.5247418e+00	 1.2742863e-01	 1.5795430e+00	 1.3619105e-01	  1.1316703e+00 	 1.8760061e-01


.. parsed-literal::

     131	 1.5260135e+00	 1.2731934e-01	 1.5807749e+00	 1.3620621e-01	  1.1390114e+00 	 2.0951962e-01
     132	 1.5269880e+00	 1.2720471e-01	 1.5817283e+00	 1.3657041e-01	  1.1531988e+00 	 1.7640877e-01


.. parsed-literal::

     133	 1.5283819e+00	 1.2717623e-01	 1.5831027e+00	 1.3648319e-01	  1.1519089e+00 	 2.0578194e-01


.. parsed-literal::

     134	 1.5290297e+00	 1.2715367e-01	 1.5837709e+00	 1.3645773e-01	  1.1505099e+00 	 2.1566844e-01


.. parsed-literal::

     135	 1.5304287e+00	 1.2706801e-01	 1.5852673e+00	 1.3639885e-01	  1.1495184e+00 	 2.2081661e-01
     136	 1.5322350e+00	 1.2687426e-01	 1.5871868e+00	 1.3630943e-01	  1.1471519e+00 	 2.0536804e-01


.. parsed-literal::

     137	 1.5335868e+00	 1.2673053e-01	 1.5886687e+00	 1.3609862e-01	  1.1473735e+00 	 3.2086253e-01


.. parsed-literal::

     138	 1.5353760e+00	 1.2646672e-01	 1.5904860e+00	 1.3610517e-01	  1.1474885e+00 	 2.1002412e-01
     139	 1.5365613e+00	 1.2631059e-01	 1.5916464e+00	 1.3611056e-01	  1.1500637e+00 	 1.9891882e-01


.. parsed-literal::

     140	 1.5377407e+00	 1.2616556e-01	 1.5927596e+00	 1.3602260e-01	  1.1565297e+00 	 1.9836855e-01


.. parsed-literal::

     141	 1.5390152e+00	 1.2602999e-01	 1.5939897e+00	 1.3578788e-01	  1.1608016e+00 	 2.1266174e-01


.. parsed-literal::

     142	 1.5402769e+00	 1.2596211e-01	 1.5952469e+00	 1.3561307e-01	  1.1660647e+00 	 2.0928931e-01


.. parsed-literal::

     143	 1.5411720e+00	 1.2594077e-01	 1.5961871e+00	 1.3543720e-01	  1.1626851e+00 	 2.0389867e-01


.. parsed-literal::

     144	 1.5425854e+00	 1.2590230e-01	 1.5977350e+00	 1.3525884e-01	  1.1536219e+00 	 2.1726465e-01


.. parsed-literal::

     145	 1.5435858e+00	 1.2581455e-01	 1.5989106e+00	 1.3492856e-01	  1.1406602e+00 	 2.0428634e-01


.. parsed-literal::

     146	 1.5451050e+00	 1.2571533e-01	 1.6004435e+00	 1.3496271e-01	  1.1390873e+00 	 2.0956135e-01
     147	 1.5458542e+00	 1.2565741e-01	 1.6011507e+00	 1.3502165e-01	  1.1404853e+00 	 1.9851160e-01


.. parsed-literal::

     148	 1.5468153e+00	 1.2559288e-01	 1.6020841e+00	 1.3502233e-01	  1.1408892e+00 	 2.0956063e-01
     149	 1.5480933e+00	 1.2559222e-01	 1.6033396e+00	 1.3510676e-01	  1.1445498e+00 	 1.7039037e-01


.. parsed-literal::

     150	 1.5495478e+00	 1.2553642e-01	 1.6047477e+00	 1.3497461e-01	  1.1409280e+00 	 2.1099353e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.13 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7efe7046f040>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.13 s, sys: 47 ms, total: 2.17 s
    Wall time: 673 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

