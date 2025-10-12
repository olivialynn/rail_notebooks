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

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4549926e-01	 3.2120108e-01	-3.3586530e-01	 3.1862963e-01	[-3.3092546e-01]	 4.5974278e-01


.. parsed-literal::

       2	-2.7530535e-01	 3.1086143e-01	-2.5161936e-01	 3.0701595e-01	[-2.4018580e-01]	 2.2726321e-01


.. parsed-literal::

       3	-2.2816685e-01	 2.8839637e-01	-1.8427642e-01	 2.8552250e-01	[-1.6963006e-01]	 2.7419496e-01


.. parsed-literal::

       4	-2.0063958e-01	 2.6574010e-01	-1.6034014e-01	 2.6326860e-01	[-1.4617824e-01]	 2.0233297e-01
       5	-1.0432593e-01	 2.5771690e-01	-6.9588812e-02	 2.5564740e-01	[-5.9313528e-02]	 2.0207214e-01


.. parsed-literal::

       6	-7.4408757e-02	 2.5260924e-01	-4.3482039e-02	 2.5182593e-01	[-4.0256073e-02]	 2.1535397e-01


.. parsed-literal::

       7	-5.4619749e-02	 2.4942317e-01	-3.0749874e-02	 2.4812326e-01	[-2.5525690e-02]	 2.0632267e-01


.. parsed-literal::

       8	-4.3250806e-02	 2.4758129e-01	-2.3043683e-02	 2.4622854e-01	[-1.7524929e-02]	 2.1379995e-01


.. parsed-literal::

       9	-2.9329120e-02	 2.4499579e-01	-1.2083869e-02	 2.4351599e-01	[-6.1213482e-03]	 2.1064782e-01


.. parsed-literal::

      10	-1.9303258e-02	 2.4288735e-01	-4.1300763e-03	 2.4115976e-01	[ 3.6020163e-03]	 2.0839810e-01


.. parsed-literal::

      11	-1.3173874e-02	 2.4210986e-01	 8.6006694e-04	 2.4008122e-01	[ 8.7813475e-03]	 2.0732951e-01
      12	-1.0463377e-02	 2.4162355e-01	 3.4257003e-03	 2.3950011e-01	[ 1.1980211e-02]	 1.9221687e-01


.. parsed-literal::

      13	-6.2200017e-03	 2.4077185e-01	 7.8122953e-03	 2.3841429e-01	[ 1.7370737e-02]	 1.9884896e-01


.. parsed-literal::

      14	 1.7199931e-01	 2.2182044e-01	 1.9981723e-01	 2.1912826e-01	[ 2.1201166e-01]	 4.2691016e-01


.. parsed-literal::

      15	 2.0336568e-01	 2.2506909e-01	 2.3177928e-01	 2.2589546e-01	[ 2.4095397e-01]	 2.9064274e-01
      16	 2.5737775e-01	 2.2076005e-01	 2.8587996e-01	 2.1860230e-01	[ 3.0352061e-01]	 1.8446708e-01


.. parsed-literal::

      17	 3.2494332e-01	 2.1256422e-01	 3.5673391e-01	 2.1232181e-01	[ 3.7619691e-01]	 2.0616484e-01


.. parsed-literal::

      18	 3.6818176e-01	 2.0827716e-01	 4.0168053e-01	 2.0816530e-01	[ 4.2797107e-01]	 2.1554661e-01


.. parsed-literal::

      19	 4.0862590e-01	 2.0471531e-01	 4.4339095e-01	 2.0229430e-01	[ 4.8093942e-01]	 2.1548533e-01
      20	 4.8001413e-01	 2.0007544e-01	 5.1623864e-01	 1.9716490e-01	[ 5.5903564e-01]	 1.9866920e-01


.. parsed-literal::

      21	 5.3305773e-01	 1.9878101e-01	 5.7017312e-01	 1.9700757e-01	[ 6.3151025e-01]	 2.0842457e-01


.. parsed-literal::

      22	 5.7244479e-01	 1.9369197e-01	 6.1045201e-01	 1.9110028e-01	[ 6.8608464e-01]	 2.0524859e-01


.. parsed-literal::

      23	 6.1020961e-01	 1.8981687e-01	 6.4841563e-01	 1.8507208e-01	[ 7.3008482e-01]	 2.0878124e-01
      24	 6.5821499e-01	 1.9014315e-01	 6.9645924e-01	 1.8418100e-01	[ 7.7310460e-01]	 2.0357370e-01


.. parsed-literal::

      25	 7.1073834e-01	 1.9562194e-01	 7.4805645e-01	 1.8934399e-01	[ 8.0245201e-01]	 1.7576790e-01
      26	 7.4937335e-01	 2.0262593e-01	 7.8767389e-01	 1.9436634e-01	[ 8.4312959e-01]	 1.8670273e-01


.. parsed-literal::

      27	 7.8529956e-01	 2.0001470e-01	 8.2533967e-01	 1.9124743e-01	[ 8.8662602e-01]	 2.1638513e-01


.. parsed-literal::

      28	 8.2028783e-01	 2.0011147e-01	 8.6174034e-01	 1.9119777e-01	[ 9.0923822e-01]	 2.0799851e-01


.. parsed-literal::

      29	 8.4499951e-01	 1.9982678e-01	 8.8691117e-01	 1.9189281e-01	[ 9.2423617e-01]	 2.2129440e-01


.. parsed-literal::

      30	 8.6625537e-01	 1.9622744e-01	 9.0774062e-01	 1.8860967e-01	[ 9.3916506e-01]	 2.0539546e-01


.. parsed-literal::

      31	 8.9473002e-01	 1.9020102e-01	 9.3634854e-01	 1.8334605e-01	[ 9.6169462e-01]	 2.1649837e-01


.. parsed-literal::

      32	 9.3041821e-01	 1.8047132e-01	 9.7237657e-01	 1.7440303e-01	[ 9.9595224e-01]	 2.0568895e-01


.. parsed-literal::

      33	 9.4874641e-01	 1.7288239e-01	 9.9249405e-01	 1.6842624e-01	[ 1.0021437e+00]	 2.0240235e-01
      34	 9.7983027e-01	 1.6977480e-01	 1.0231437e+00	 1.6521496e-01	[ 1.0394470e+00]	 1.9735193e-01


.. parsed-literal::

      35	 9.9802748e-01	 1.6643117e-01	 1.0417692e+00	 1.6146000e-01	[ 1.0605200e+00]	 2.1505427e-01


.. parsed-literal::

      36	 1.0168497e+00	 1.6213885e-01	 1.0618094e+00	 1.5775258e-01	[ 1.0755995e+00]	 2.1733856e-01


.. parsed-literal::

      37	 1.0310624e+00	 1.5751284e-01	 1.0768075e+00	 1.5328759e-01	[ 1.0906805e+00]	 2.0639467e-01
      38	 1.0456789e+00	 1.5593735e-01	 1.0914976e+00	 1.5222806e-01	[ 1.1024267e+00]	 1.7455626e-01


.. parsed-literal::

      39	 1.0577727e+00	 1.5401238e-01	 1.1038264e+00	 1.5032402e-01	[ 1.1104180e+00]	 1.9429278e-01


.. parsed-literal::

      40	 1.0764665e+00	 1.5034364e-01	 1.1230046e+00	 1.4579119e-01	[ 1.1260890e+00]	 2.0792627e-01


.. parsed-literal::

      41	 1.0845179e+00	 1.4643496e-01	 1.1319186e+00	 1.4215923e-01	[ 1.1271249e+00]	 2.0367360e-01


.. parsed-literal::

      42	 1.1018470e+00	 1.4558493e-01	 1.1485159e+00	 1.4065313e-01	[ 1.1485691e+00]	 2.0806527e-01


.. parsed-literal::

      43	 1.1102103e+00	 1.4463520e-01	 1.1570508e+00	 1.3923508e-01	[ 1.1605627e+00]	 2.0262218e-01


.. parsed-literal::

      44	 1.1223474e+00	 1.4359364e-01	 1.1692044e+00	 1.3795289e-01	[ 1.1772596e+00]	 2.0246649e-01


.. parsed-literal::

      45	 1.1359330e+00	 1.4288429e-01	 1.1827382e+00	 1.3652563e-01	[ 1.1861891e+00]	 2.0528007e-01


.. parsed-literal::

      46	 1.1483465e+00	 1.4191678e-01	 1.1953126e+00	 1.3633749e-01	[ 1.1988346e+00]	 2.1684813e-01


.. parsed-literal::

      47	 1.1583603e+00	 1.4152127e-01	 1.2052539e+00	 1.3668739e-01	[ 1.2042468e+00]	 2.0119572e-01
      48	 1.1708745e+00	 1.4032026e-01	 1.2179937e+00	 1.3581525e-01	[ 1.2094341e+00]	 2.0014238e-01


.. parsed-literal::

      49	 1.1816808e+00	 1.3972212e-01	 1.2288335e+00	 1.3571081e-01	[ 1.2189012e+00]	 2.1611023e-01


.. parsed-literal::

      50	 1.1916951e+00	 1.3882500e-01	 1.2387694e+00	 1.3519744e-01	[ 1.2255038e+00]	 2.0678997e-01


.. parsed-literal::

      51	 1.2048599e+00	 1.3764461e-01	 1.2521496e+00	 1.3402165e-01	[ 1.2339875e+00]	 2.0124269e-01


.. parsed-literal::

      52	 1.2142053e+00	 1.3726738e-01	 1.2616903e+00	 1.3338634e-01	[ 1.2380537e+00]	 2.0518398e-01
      53	 1.2253879e+00	 1.3680648e-01	 1.2730836e+00	 1.3229906e-01	[ 1.2459585e+00]	 1.6424632e-01


.. parsed-literal::

      54	 1.2336635e+00	 1.3608838e-01	 1.2815503e+00	 1.3081570e-01	  1.2450125e+00 	 1.9906020e-01


.. parsed-literal::

      55	 1.2439187e+00	 1.3511225e-01	 1.2921448e+00	 1.2914273e-01	  1.2445362e+00 	 2.1574211e-01
      56	 1.2560838e+00	 1.3395671e-01	 1.3044385e+00	 1.2705979e-01	  1.2431501e+00 	 1.9742322e-01


.. parsed-literal::

      57	 1.2665032e+00	 1.3317349e-01	 1.3149461e+00	 1.2553430e-01	[ 1.2491958e+00]	 2.0956254e-01
      58	 1.2747701e+00	 1.3260337e-01	 1.3232353e+00	 1.2439087e-01	[ 1.2633601e+00]	 2.0194197e-01


.. parsed-literal::

      59	 1.2818055e+00	 1.3206119e-01	 1.3303846e+00	 1.2339795e-01	[ 1.2706928e+00]	 1.9687963e-01


.. parsed-literal::

      60	 1.2883459e+00	 1.3167688e-01	 1.3371762e+00	 1.2230090e-01	[ 1.2829523e+00]	 2.0131731e-01
      61	 1.2942952e+00	 1.3125730e-01	 1.3432718e+00	 1.2156572e-01	[ 1.2886530e+00]	 2.0350790e-01


.. parsed-literal::

      62	 1.3031520e+00	 1.3073084e-01	 1.3523067e+00	 1.2073443e-01	[ 1.2985687e+00]	 2.1524334e-01


.. parsed-literal::

      63	 1.3090418e+00	 1.3055503e-01	 1.3584390e+00	 1.1948214e-01	[ 1.3095627e+00]	 2.0587039e-01


.. parsed-literal::

      64	 1.3165430e+00	 1.3027323e-01	 1.3656386e+00	 1.1917978e-01	[ 1.3164589e+00]	 2.1623039e-01
      65	 1.3226290e+00	 1.3009790e-01	 1.3715796e+00	 1.1867807e-01	[ 1.3228409e+00]	 2.0009494e-01


.. parsed-literal::

      66	 1.3273110e+00	 1.3019209e-01	 1.3761971e+00	 1.1888045e-01	[ 1.3232016e+00]	 2.1519089e-01
      67	 1.3344247e+00	 1.3023239e-01	 1.3835287e+00	 1.1904129e-01	  1.3224456e+00 	 1.8037319e-01


.. parsed-literal::

      68	 1.3401795e+00	 1.3055468e-01	 1.3895419e+00	 1.1920629e-01	  1.3145770e+00 	 2.2022319e-01


.. parsed-literal::

      69	 1.3455597e+00	 1.3039756e-01	 1.3949555e+00	 1.1930546e-01	  1.3134224e+00 	 2.1545100e-01


.. parsed-literal::

      70	 1.3503801e+00	 1.3014939e-01	 1.3998656e+00	 1.1906362e-01	  1.3122710e+00 	 2.1022391e-01


.. parsed-literal::

      71	 1.3575679e+00	 1.2989670e-01	 1.4071902e+00	 1.1895137e-01	  1.3063314e+00 	 2.1946907e-01


.. parsed-literal::

      72	 1.3609009e+00	 1.2944727e-01	 1.4112822e+00	 1.1890333e-01	  1.2876220e+00 	 2.0896673e-01


.. parsed-literal::

      73	 1.3692061e+00	 1.2932741e-01	 1.4191852e+00	 1.1868790e-01	  1.2948441e+00 	 2.1881437e-01


.. parsed-literal::

      74	 1.3718327e+00	 1.2922764e-01	 1.4218050e+00	 1.1855824e-01	  1.2979761e+00 	 2.0980883e-01


.. parsed-literal::

      75	 1.3766865e+00	 1.2899586e-01	 1.4268505e+00	 1.1861111e-01	  1.2921002e+00 	 2.1248198e-01


.. parsed-literal::

      76	 1.3796663e+00	 1.2860309e-01	 1.4302563e+00	 1.1861518e-01	  1.2823891e+00 	 2.1546602e-01


.. parsed-literal::

      77	 1.3845564e+00	 1.2848201e-01	 1.4350650e+00	 1.1874390e-01	  1.2777312e+00 	 2.2079659e-01
      78	 1.3870929e+00	 1.2829291e-01	 1.4376394e+00	 1.1869053e-01	  1.2764029e+00 	 1.8955350e-01


.. parsed-literal::

      79	 1.3893466e+00	 1.2812091e-01	 1.4399553e+00	 1.1861838e-01	  1.2751454e+00 	 2.1448421e-01


.. parsed-literal::

      80	 1.3938046e+00	 1.2782877e-01	 1.4445139e+00	 1.1841659e-01	  1.2722178e+00 	 2.1808720e-01
      81	 1.3947155e+00	 1.2774485e-01	 1.4458350e+00	 1.1841668e-01	  1.2550470e+00 	 1.8466377e-01


.. parsed-literal::

      82	 1.4004760e+00	 1.2740728e-01	 1.4513165e+00	 1.1802501e-01	  1.2667140e+00 	 2.0550942e-01
      83	 1.4027397e+00	 1.2724532e-01	 1.4535595e+00	 1.1791220e-01	  1.2649112e+00 	 1.8503237e-01


.. parsed-literal::

      84	 1.4059215e+00	 1.2695341e-01	 1.4568164e+00	 1.1776609e-01	  1.2632334e+00 	 2.0968485e-01
      85	 1.4080180e+00	 1.2686156e-01	 1.4589944e+00	 1.1783147e-01	  1.2485433e+00 	 1.8625689e-01


.. parsed-literal::

      86	 1.4109902e+00	 1.2671373e-01	 1.4619657e+00	 1.1773843e-01	  1.2528918e+00 	 1.9926119e-01


.. parsed-literal::

      87	 1.4129484e+00	 1.2652695e-01	 1.4639204e+00	 1.1756188e-01	  1.2578635e+00 	 2.0690846e-01


.. parsed-literal::

      88	 1.4149235e+00	 1.2637282e-01	 1.4659095e+00	 1.1736215e-01	  1.2600868e+00 	 2.1386075e-01
      89	 1.4183430e+00	 1.2615943e-01	 1.4694284e+00	 1.1710271e-01	  1.2611539e+00 	 1.9858479e-01


.. parsed-literal::

      90	 1.4203127e+00	 1.2619318e-01	 1.4715181e+00	 1.1705255e-01	  1.2577119e+00 	 1.9770169e-01
      91	 1.4226479e+00	 1.2606534e-01	 1.4737637e+00	 1.1698800e-01	  1.2576535e+00 	 1.9726276e-01


.. parsed-literal::

      92	 1.4243450e+00	 1.2595147e-01	 1.4754446e+00	 1.1700434e-01	  1.2540523e+00 	 2.0578098e-01
      93	 1.4267983e+00	 1.2573391e-01	 1.4779105e+00	 1.1705713e-01	  1.2510746e+00 	 1.9878840e-01


.. parsed-literal::

      94	 1.4289359e+00	 1.2539707e-01	 1.4802640e+00	 1.1764491e-01	  1.2268435e+00 	 1.8945193e-01
      95	 1.4326900e+00	 1.2525991e-01	 1.4838632e+00	 1.1761469e-01	  1.2351839e+00 	 1.7908406e-01


.. parsed-literal::

      96	 1.4343334e+00	 1.2518017e-01	 1.4854648e+00	 1.1765374e-01	  1.2397816e+00 	 1.7763066e-01


.. parsed-literal::

      97	 1.4362544e+00	 1.2505002e-01	 1.4873792e+00	 1.1778606e-01	  1.2407365e+00 	 2.1014643e-01
      98	 1.4377092e+00	 1.2476598e-01	 1.4889329e+00	 1.1788954e-01	  1.2502157e+00 	 1.9544649e-01


.. parsed-literal::

      99	 1.4402922e+00	 1.2467731e-01	 1.4914587e+00	 1.1794308e-01	  1.2443395e+00 	 2.0567155e-01
     100	 1.4423588e+00	 1.2453948e-01	 1.4935373e+00	 1.1786854e-01	  1.2390074e+00 	 2.0007610e-01


.. parsed-literal::

     101	 1.4441585e+00	 1.2439086e-01	 1.4953775e+00	 1.1781479e-01	  1.2343796e+00 	 2.1029282e-01
     102	 1.4479523e+00	 1.2407206e-01	 1.4992268e+00	 1.1774747e-01	  1.2190071e+00 	 1.7841363e-01


.. parsed-literal::

     103	 1.4496535e+00	 1.2397062e-01	 1.5009917e+00	 1.1783650e-01	  1.2109673e+00 	 3.2145047e-01


.. parsed-literal::

     104	 1.4512645e+00	 1.2389516e-01	 1.5025880e+00	 1.1794944e-01	  1.2085259e+00 	 2.1467018e-01
     105	 1.4527481e+00	 1.2382266e-01	 1.5040724e+00	 1.1809011e-01	  1.2063022e+00 	 2.0788026e-01


.. parsed-literal::

     106	 1.4553937e+00	 1.2357500e-01	 1.5067883e+00	 1.1822086e-01	  1.2042205e+00 	 1.8022823e-01
     107	 1.4574964e+00	 1.2327755e-01	 1.5090859e+00	 1.1852709e-01	  1.1988517e+00 	 1.9346881e-01


.. parsed-literal::

     108	 1.4600456e+00	 1.2312817e-01	 1.5116051e+00	 1.1834973e-01	  1.1953416e+00 	 2.0975280e-01
     109	 1.4616509e+00	 1.2302413e-01	 1.5132127e+00	 1.1807447e-01	  1.1992300e+00 	 2.0133233e-01


.. parsed-literal::

     110	 1.4639161e+00	 1.2282984e-01	 1.5156146e+00	 1.1779908e-01	  1.2037430e+00 	 2.0566249e-01
     111	 1.4656394e+00	 1.2279866e-01	 1.5174254e+00	 1.1735186e-01	  1.2079538e+00 	 1.9923043e-01


.. parsed-literal::

     112	 1.4670124e+00	 1.2276941e-01	 1.5187448e+00	 1.1735908e-01	  1.2123241e+00 	 2.0169735e-01


.. parsed-literal::

     113	 1.4681953e+00	 1.2276667e-01	 1.5199186e+00	 1.1736863e-01	  1.2137536e+00 	 2.0492864e-01


.. parsed-literal::

     114	 1.4695716e+00	 1.2278182e-01	 1.5212948e+00	 1.1725895e-01	  1.2149306e+00 	 2.0572829e-01
     115	 1.4724448e+00	 1.2282096e-01	 1.5242147e+00	 1.1698720e-01	  1.2180238e+00 	 1.9647145e-01


.. parsed-literal::

     116	 1.4729072e+00	 1.2288677e-01	 1.5248120e+00	 1.1627216e-01	  1.2271850e+00 	 2.1489382e-01
     117	 1.4751754e+00	 1.2279861e-01	 1.5269647e+00	 1.1632366e-01	  1.2263417e+00 	 1.8049717e-01


.. parsed-literal::

     118	 1.4759776e+00	 1.2276227e-01	 1.5277562e+00	 1.1630588e-01	  1.2268635e+00 	 2.1503711e-01
     119	 1.4771049e+00	 1.2271334e-01	 1.5288811e+00	 1.1616802e-01	  1.2301259e+00 	 1.9877934e-01


.. parsed-literal::

     120	 1.4796179e+00	 1.2273226e-01	 1.5313929e+00	 1.1594375e-01	  1.2320520e+00 	 1.8332458e-01


.. parsed-literal::

     121	 1.4813932e+00	 1.2285946e-01	 1.5332354e+00	 1.1567784e-01	  1.2473459e+00 	 2.1324182e-01


.. parsed-literal::

     122	 1.4828234e+00	 1.2285142e-01	 1.5346370e+00	 1.1565921e-01	  1.2462940e+00 	 2.1178627e-01


.. parsed-literal::

     123	 1.4844864e+00	 1.2285501e-01	 1.5362982e+00	 1.1563914e-01	  1.2500875e+00 	 2.1043801e-01
     124	 1.4849093e+00	 1.2289919e-01	 1.5367990e+00	 1.1555335e-01	  1.2483045e+00 	 1.8368673e-01


.. parsed-literal::

     125	 1.4865333e+00	 1.2282275e-01	 1.5383651e+00	 1.1553548e-01	  1.2538642e+00 	 2.1625662e-01


.. parsed-literal::

     126	 1.4875580e+00	 1.2275071e-01	 1.5393812e+00	 1.1545395e-01	  1.2579232e+00 	 2.1827316e-01
     127	 1.4887116e+00	 1.2268147e-01	 1.5405650e+00	 1.1529544e-01	  1.2625214e+00 	 2.0119190e-01


.. parsed-literal::

     128	 1.4903860e+00	 1.2257991e-01	 1.5422800e+00	 1.1506150e-01	  1.2654649e+00 	 2.0010638e-01


.. parsed-literal::

     129	 1.4917092e+00	 1.2251105e-01	 1.5436844e+00	 1.1481670e-01	  1.2661104e+00 	 3.0646205e-01


.. parsed-literal::

     130	 1.4932013e+00	 1.2243888e-01	 1.5451787e+00	 1.1465333e-01	  1.2648934e+00 	 2.1376538e-01


.. parsed-literal::

     131	 1.4940699e+00	 1.2241320e-01	 1.5460490e+00	 1.1457693e-01	  1.2645854e+00 	 2.0749021e-01


.. parsed-literal::

     132	 1.4954862e+00	 1.2235772e-01	 1.5474809e+00	 1.1444568e-01	  1.2665251e+00 	 2.0883107e-01


.. parsed-literal::

     133	 1.4963463e+00	 1.2229571e-01	 1.5483787e+00	 1.1430470e-01	  1.2740066e+00 	 3.2090306e-01


.. parsed-literal::

     134	 1.4976830e+00	 1.2221809e-01	 1.5497231e+00	 1.1415964e-01	  1.2788018e+00 	 2.0525265e-01
     135	 1.4986718e+00	 1.2214555e-01	 1.5507129e+00	 1.1411508e-01	  1.2845153e+00 	 1.7395473e-01


.. parsed-literal::

     136	 1.4998155e+00	 1.2205515e-01	 1.5518671e+00	 1.1397684e-01	  1.2918149e+00 	 1.9177675e-01


.. parsed-literal::

     137	 1.5010141e+00	 1.2195444e-01	 1.5530761e+00	 1.1392587e-01	  1.2953245e+00 	 2.5582433e-01


.. parsed-literal::

     138	 1.5022946e+00	 1.2185265e-01	 1.5543601e+00	 1.1376083e-01	  1.2994780e+00 	 2.1403956e-01
     139	 1.5034311e+00	 1.2173094e-01	 1.5555358e+00	 1.1346293e-01	  1.2907566e+00 	 1.8995309e-01


.. parsed-literal::

     140	 1.5045035e+00	 1.2166150e-01	 1.5566178e+00	 1.1332850e-01	  1.2918563e+00 	 2.1706557e-01


.. parsed-literal::

     141	 1.5053690e+00	 1.2159384e-01	 1.5575150e+00	 1.1313267e-01	  1.2918912e+00 	 2.0577669e-01


.. parsed-literal::

     142	 1.5062457e+00	 1.2148319e-01	 1.5584589e+00	 1.1287329e-01	  1.2877110e+00 	 2.1521688e-01


.. parsed-literal::

     143	 1.5071903e+00	 1.2136020e-01	 1.5594629e+00	 1.1254973e-01	  1.2861795e+00 	 2.1008921e-01
     144	 1.5081697e+00	 1.2127706e-01	 1.5604664e+00	 1.1231603e-01	  1.2829060e+00 	 1.9781661e-01


.. parsed-literal::

     145	 1.5097337e+00	 1.2109301e-01	 1.5620871e+00	 1.1197883e-01	  1.2708454e+00 	 2.0585418e-01


.. parsed-literal::

     146	 1.5108081e+00	 1.2103823e-01	 1.5631664e+00	 1.1170097e-01	  1.2613077e+00 	 2.1894503e-01


.. parsed-literal::

     147	 1.5118683e+00	 1.2096896e-01	 1.5642017e+00	 1.1162843e-01	  1.2557867e+00 	 2.0828104e-01


.. parsed-literal::

     148	 1.5127598e+00	 1.2092565e-01	 1.5650713e+00	 1.1162649e-01	  1.2491650e+00 	 2.1477890e-01


.. parsed-literal::

     149	 1.5134683e+00	 1.2088260e-01	 1.5657760e+00	 1.1159889e-01	  1.2428561e+00 	 2.1019626e-01


.. parsed-literal::

     150	 1.5142884e+00	 1.2080093e-01	 1.5666262e+00	 1.1145139e-01	  1.2372809e+00 	 2.0711136e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.11 s, total: 2min 6s
    Wall time: 31.9 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f02fc0a5b70>



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
    CPU times: user 1.81 s, sys: 32 ms, total: 1.85 s
    Wall time: 587 ms


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

