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
       1	-3.3702993e-01	 3.1829668e-01	-3.2728855e-01	 3.2981555e-01	[-3.4907846e-01]	 4.6577191e-01


.. parsed-literal::

       2	-2.6406504e-01	 3.0703661e-01	-2.3963615e-01	 3.1843449e-01	[-2.7513501e-01]	 2.3705912e-01


.. parsed-literal::

       3	-2.2054221e-01	 2.8790994e-01	-1.8012622e-01	 2.9888257e-01	[-2.2775675e-01]	 2.8291416e-01


.. parsed-literal::

       4	-1.9173755e-01	 2.6326574e-01	-1.5027430e-01	 2.7640632e-01	 -2.3099777e-01 	 2.0188260e-01
       5	-1.0234239e-01	 2.5572569e-01	-6.5391809e-02	 2.7061662e-01	[-1.4103790e-01]	 2.1105647e-01


.. parsed-literal::

       6	-6.3851569e-02	 2.4903251e-01	-2.9782822e-02	 2.6340792e-01	[-8.5913529e-02]	 2.1458125e-01


.. parsed-literal::

       7	-4.6180679e-02	 2.4662367e-01	-1.9314377e-02	 2.6101412e-01	[-7.9315307e-02]	 2.0892286e-01


.. parsed-literal::

       8	-2.9816206e-02	 2.4388049e-01	-7.8341954e-03	 2.5809692e-01	[-6.8267767e-02]	 2.0887160e-01


.. parsed-literal::

       9	-1.1903986e-02	 2.4056340e-01	 6.3608709e-03	 2.5571779e-01	[-5.9849925e-02]	 2.1953559e-01


.. parsed-literal::

      10	-1.4074014e-03	 2.3893147e-01	 1.3861095e-02	 2.5668713e-01	 -6.2023838e-02 	 2.0492315e-01


.. parsed-literal::

      11	 5.1912894e-03	 2.3759242e-01	 1.9598804e-02	 2.5663464e-01	 -6.5530760e-02 	 2.0682383e-01


.. parsed-literal::

      12	 7.7051632e-03	 2.3711767e-01	 2.1917043e-02	 2.5598651e-01	 -5.9882737e-02 	 2.1088648e-01


.. parsed-literal::

      13	 1.1531591e-02	 2.3643395e-01	 2.5445790e-02	 2.5506396e-01	[-5.5565122e-02]	 2.0403671e-01


.. parsed-literal::

      14	 1.7520148e-02	 2.3499943e-01	 3.2354624e-02	 2.5419162e-01	[-4.6326237e-02]	 2.1138954e-01
      15	 5.7561553e-02	 2.2797115e-01	 7.3371371e-02	 2.4532303e-01	[ 1.2769112e-02]	 2.0225883e-01


.. parsed-literal::

      16	 8.8702595e-02	 2.2204640e-01	 1.0771977e-01	 2.3671944e-01	[ 6.8194612e-02]	 3.2366395e-01


.. parsed-literal::

      17	 1.4550640e-01	 2.1641485e-01	 1.6632201e-01	 2.3381473e-01	[ 1.2394905e-01]	 2.1905375e-01


.. parsed-literal::

      18	 2.6793468e-01	 2.1386705e-01	 2.9430270e-01	 2.3037226e-01	[ 2.1597364e-01]	 2.1965027e-01
      19	 3.1826637e-01	 2.1036144e-01	 3.4741189e-01	 2.2947466e-01	[ 2.3593843e-01]	 1.8987489e-01


.. parsed-literal::

      20	 3.9004734e-01	 2.0221022e-01	 4.2300494e-01	 2.2565486e-01	[ 2.8975982e-01]	 1.7804670e-01


.. parsed-literal::

      21	 4.4233607e-01	 2.0764633e-01	 4.7729931e-01	 2.2574081e-01	[ 3.7016626e-01]	 2.1384144e-01


.. parsed-literal::

      22	 5.1890718e-01	 2.0058543e-01	 5.5391050e-01	 2.2154585e-01	[ 4.5531209e-01]	 2.1140575e-01


.. parsed-literal::

      23	 5.7684378e-01	 1.9304090e-01	 6.1250209e-01	 2.1507454e-01	[ 5.1465228e-01]	 2.1198606e-01


.. parsed-literal::

      24	 6.2980291e-01	 1.8991969e-01	 6.6735890e-01	 2.0966260e-01	[ 5.8511393e-01]	 2.1704984e-01


.. parsed-literal::

      25	 6.7182036e-01	 1.8619264e-01	 7.1049543e-01	 2.0395165e-01	[ 6.5236781e-01]	 2.1688318e-01
      26	 6.9984265e-01	 1.8303892e-01	 7.3665996e-01	 1.9982028e-01	[ 6.9171839e-01]	 1.8177772e-01


.. parsed-literal::

      27	 7.1925627e-01	 1.8384344e-01	 7.5593986e-01	 1.9845370e-01	[ 7.1557852e-01]	 2.0987964e-01


.. parsed-literal::

      28	 7.4299356e-01	 1.8677648e-01	 7.7974845e-01	 1.9907927e-01	[ 7.4018223e-01]	 3.0516768e-01


.. parsed-literal::

      29	 7.5726444e-01	 1.9300246e-01	 7.9469597e-01	 2.0379567e-01	[ 7.4986058e-01]	 2.1740937e-01


.. parsed-literal::

      30	 7.8666111e-01	 1.8533680e-01	 8.2462303e-01	 1.9849187e-01	[ 7.7363499e-01]	 2.0879889e-01


.. parsed-literal::

      31	 8.0848482e-01	 1.8092760e-01	 8.4666013e-01	 1.9538343e-01	[ 7.9106338e-01]	 2.0387602e-01


.. parsed-literal::

      32	 8.3029220e-01	 1.7690591e-01	 8.6904203e-01	 1.9240322e-01	[ 8.1132509e-01]	 2.0653987e-01


.. parsed-literal::

      33	 8.4741123e-01	 1.7820863e-01	 8.8640518e-01	 1.9521234e-01	[ 8.1419264e-01]	 2.0452237e-01


.. parsed-literal::

      34	 8.6725283e-01	 1.7691968e-01	 9.0644107e-01	 1.9419461e-01	[ 8.3619180e-01]	 2.0704699e-01


.. parsed-literal::

      35	 8.8858939e-01	 1.7717919e-01	 9.2833944e-01	 1.9483139e-01	[ 8.5866377e-01]	 2.1654987e-01


.. parsed-literal::

      36	 9.0512193e-01	 1.7681606e-01	 9.4559589e-01	 1.9536990e-01	[ 8.7419730e-01]	 2.1151090e-01


.. parsed-literal::

      37	 9.2328562e-01	 1.7462380e-01	 9.6456051e-01	 1.9380520e-01	[ 8.8690637e-01]	 2.0907831e-01


.. parsed-literal::

      38	 9.3647826e-01	 1.7357411e-01	 9.7820983e-01	 1.9336786e-01	[ 8.9707656e-01]	 2.1484137e-01


.. parsed-literal::

      39	 9.6352596e-01	 1.7218180e-01	 1.0062271e+00	 1.9226688e-01	[ 9.2112974e-01]	 2.1720791e-01


.. parsed-literal::

      40	 9.6861475e-01	 1.7476878e-01	 1.0129428e+00	 1.9499632e-01	  9.1961981e-01 	 2.1038365e-01


.. parsed-literal::

      41	 9.9489073e-01	 1.7200744e-01	 1.0387209e+00	 1.9357567e-01	[ 9.3629997e-01]	 2.0726109e-01


.. parsed-literal::

      42	 1.0071217e+00	 1.7097442e-01	 1.0507707e+00	 1.9269669e-01	[ 9.4430950e-01]	 2.1463561e-01


.. parsed-literal::

      43	 1.0207722e+00	 1.6993343e-01	 1.0646207e+00	 1.9216332e-01	[ 9.5432131e-01]	 2.6831341e-01


.. parsed-literal::

      44	 1.0366729e+00	 1.6886815e-01	 1.0814602e+00	 1.9190309e-01	  9.5217105e-01 	 2.1043181e-01


.. parsed-literal::

      45	 1.0497973e+00	 1.6751439e-01	 1.0947243e+00	 1.9184353e-01	[ 9.6721120e-01]	 2.1657753e-01


.. parsed-literal::

      46	 1.0619333e+00	 1.6683489e-01	 1.1068507e+00	 1.9173539e-01	[ 9.7986205e-01]	 2.1242404e-01


.. parsed-literal::

      47	 1.0738153e+00	 1.6450948e-01	 1.1192211e+00	 1.8962948e-01	[ 9.8835514e-01]	 2.2129965e-01
      48	 1.0853589e+00	 1.6226612e-01	 1.1314869e+00	 1.8724832e-01	[ 1.0016075e+00]	 1.8213344e-01


.. parsed-literal::

      49	 1.0957618e+00	 1.5950968e-01	 1.1422365e+00	 1.8437166e-01	[ 1.0073831e+00]	 2.1288872e-01


.. parsed-literal::

      50	 1.1046155e+00	 1.5848811e-01	 1.1510992e+00	 1.8322517e-01	[ 1.0141197e+00]	 2.1788239e-01


.. parsed-literal::

      51	 1.1140861e+00	 1.5795719e-01	 1.1606043e+00	 1.8271848e-01	[ 1.0203049e+00]	 2.2259569e-01


.. parsed-literal::

      52	 1.1245974e+00	 1.5748876e-01	 1.1712813e+00	 1.8335093e-01	[ 1.0288182e+00]	 2.1302032e-01


.. parsed-literal::

      53	 1.1336639e+00	 1.5653669e-01	 1.1803133e+00	 1.8283907e-01	[ 1.0391669e+00]	 2.0712185e-01


.. parsed-literal::

      54	 1.1399651e+00	 1.5582517e-01	 1.1867169e+00	 1.8295206e-01	[ 1.0418685e+00]	 2.1212411e-01


.. parsed-literal::

      55	 1.1498640e+00	 1.5418256e-01	 1.1971138e+00	 1.8250021e-01	[ 1.0459072e+00]	 2.1683002e-01


.. parsed-literal::

      56	 1.1583605e+00	 1.5224480e-01	 1.2059255e+00	 1.8076135e-01	  1.0458842e+00 	 2.1346211e-01


.. parsed-literal::

      57	 1.1679373e+00	 1.5099775e-01	 1.2155651e+00	 1.7914385e-01	[ 1.0488184e+00]	 2.0951390e-01


.. parsed-literal::

      58	 1.1752926e+00	 1.4944804e-01	 1.2232361e+00	 1.7698769e-01	[ 1.0500480e+00]	 2.2406530e-01


.. parsed-literal::

      59	 1.1820646e+00	 1.4918846e-01	 1.2300659e+00	 1.7597954e-01	[ 1.0541319e+00]	 2.1420360e-01


.. parsed-literal::

      60	 1.1893242e+00	 1.4850311e-01	 1.2375270e+00	 1.7489742e-01	[ 1.0593260e+00]	 2.2011042e-01


.. parsed-literal::

      61	 1.1974164e+00	 1.4719672e-01	 1.2460997e+00	 1.7375769e-01	  1.0534893e+00 	 2.1518946e-01
      62	 1.2045143e+00	 1.4645605e-01	 1.2533141e+00	 1.7341756e-01	  1.0567604e+00 	 1.9898057e-01


.. parsed-literal::

      63	 1.2111790e+00	 1.4513469e-01	 1.2600795e+00	 1.7238694e-01	  1.0589163e+00 	 2.0720577e-01


.. parsed-literal::

      64	 1.2187594e+00	 1.4344268e-01	 1.2679737e+00	 1.7127394e-01	  1.0554576e+00 	 2.2939372e-01


.. parsed-literal::

      65	 1.2255582e+00	 1.4232975e-01	 1.2750192e+00	 1.7002497e-01	[ 1.0627226e+00]	 2.0978451e-01


.. parsed-literal::

      66	 1.2315157e+00	 1.4191627e-01	 1.2807971e+00	 1.6964987e-01	[ 1.0659855e+00]	 2.0579386e-01


.. parsed-literal::

      67	 1.2375974e+00	 1.4139531e-01	 1.2870073e+00	 1.6932256e-01	  1.0633547e+00 	 2.1670914e-01


.. parsed-literal::

      68	 1.2434833e+00	 1.4105393e-01	 1.2928128e+00	 1.6928204e-01	  1.0600742e+00 	 2.1145415e-01


.. parsed-literal::

      69	 1.2490544e+00	 1.4080990e-01	 1.2987161e+00	 1.6975354e-01	  1.0325616e+00 	 2.3131776e-01


.. parsed-literal::

      70	 1.2561504e+00	 1.4032126e-01	 1.3057591e+00	 1.6970833e-01	  1.0360079e+00 	 2.0648313e-01


.. parsed-literal::

      71	 1.2597285e+00	 1.3990363e-01	 1.3093430e+00	 1.6936547e-01	  1.0399856e+00 	 2.0583677e-01
      72	 1.2655577e+00	 1.3949066e-01	 1.3154342e+00	 1.6919758e-01	  1.0390852e+00 	 1.8577099e-01


.. parsed-literal::

      73	 1.2709729e+00	 1.3880558e-01	 1.3213356e+00	 1.6856779e-01	  1.0285975e+00 	 2.1606302e-01


.. parsed-literal::

      74	 1.2772222e+00	 1.3895554e-01	 1.3274865e+00	 1.6854548e-01	  1.0351328e+00 	 2.2212148e-01


.. parsed-literal::

      75	 1.2824356e+00	 1.3928804e-01	 1.3327820e+00	 1.6864817e-01	  1.0381696e+00 	 2.1979594e-01
      76	 1.2873427e+00	 1.3953447e-01	 1.3377982e+00	 1.6853067e-01	  1.0378025e+00 	 1.7989469e-01


.. parsed-literal::

      77	 1.2927637e+00	 1.4046568e-01	 1.3435606e+00	 1.6875324e-01	  1.0455825e+00 	 2.0785856e-01


.. parsed-literal::

      78	 1.2978458e+00	 1.4009863e-01	 1.3487681e+00	 1.6845284e-01	  1.0385833e+00 	 2.1785927e-01


.. parsed-literal::

      79	 1.3011967e+00	 1.3969819e-01	 1.3520203e+00	 1.6803484e-01	  1.0414362e+00 	 2.2821474e-01


.. parsed-literal::

      80	 1.3064494e+00	 1.3928266e-01	 1.3573958e+00	 1.6789092e-01	  1.0385997e+00 	 2.1466279e-01


.. parsed-literal::

      81	 1.3117686e+00	 1.3930889e-01	 1.3629988e+00	 1.6839208e-01	  1.0396041e+00 	 2.1109056e-01
      82	 1.3170262e+00	 1.3943342e-01	 1.3681831e+00	 1.6879051e-01	  1.0449785e+00 	 1.9883013e-01


.. parsed-literal::

      83	 1.3211234e+00	 1.3959028e-01	 1.3722370e+00	 1.6918533e-01	  1.0511922e+00 	 2.1704388e-01


.. parsed-literal::

      84	 1.3256839e+00	 1.3961789e-01	 1.3768082e+00	 1.6942343e-01	  1.0571525e+00 	 2.2213912e-01


.. parsed-literal::

      85	 1.3298159e+00	 1.3944240e-01	 1.3812622e+00	 1.6968366e-01	  1.0553169e+00 	 2.1830535e-01


.. parsed-literal::

      86	 1.3359300e+00	 1.3927995e-01	 1.3872515e+00	 1.6931685e-01	  1.0640279e+00 	 2.1692228e-01


.. parsed-literal::

      87	 1.3396804e+00	 1.3901495e-01	 1.3910577e+00	 1.6871613e-01	[ 1.0674562e+00]	 2.1826506e-01


.. parsed-literal::

      88	 1.3446177e+00	 1.3845397e-01	 1.3960818e+00	 1.6745661e-01	[ 1.0721723e+00]	 2.2805119e-01


.. parsed-literal::

      89	 1.3498820e+00	 1.3791578e-01	 1.4016072e+00	 1.6653364e-01	[ 1.0857209e+00]	 2.1772146e-01


.. parsed-literal::

      90	 1.3555718e+00	 1.3731741e-01	 1.4072393e+00	 1.6525076e-01	[ 1.0936562e+00]	 2.1753955e-01
      91	 1.3593763e+00	 1.3705074e-01	 1.4110085e+00	 1.6512409e-01	[ 1.0971470e+00]	 1.9404221e-01


.. parsed-literal::

      92	 1.3632118e+00	 1.3666953e-01	 1.4149783e+00	 1.6511676e-01	  1.0964420e+00 	 2.1591139e-01


.. parsed-literal::

      93	 1.3671199e+00	 1.3628840e-01	 1.4189959e+00	 1.6504109e-01	  1.0919106e+00 	 2.2583342e-01


.. parsed-literal::

      94	 1.3716926e+00	 1.3578051e-01	 1.4237383e+00	 1.6468063e-01	  1.0855286e+00 	 2.1435118e-01


.. parsed-literal::

      95	 1.3754588e+00	 1.3514639e-01	 1.4277905e+00	 1.6375053e-01	  1.0784644e+00 	 2.1800709e-01


.. parsed-literal::

      96	 1.3800925e+00	 1.3491870e-01	 1.4323364e+00	 1.6340766e-01	  1.0774989e+00 	 2.1171260e-01


.. parsed-literal::

      97	 1.3834164e+00	 1.3476559e-01	 1.4356021e+00	 1.6304216e-01	  1.0804233e+00 	 2.2692585e-01


.. parsed-literal::

      98	 1.3886284e+00	 1.3439585e-01	 1.4408869e+00	 1.6238871e-01	  1.0799299e+00 	 2.1527123e-01


.. parsed-literal::

      99	 1.3950059e+00	 1.3400648e-01	 1.4474363e+00	 1.6148430e-01	  1.0834782e+00 	 2.2348452e-01


.. parsed-literal::

     100	 1.3979555e+00	 1.3361095e-01	 1.4505187e+00	 1.6103151e-01	  1.0756682e+00 	 3.3817959e-01


.. parsed-literal::

     101	 1.4009467e+00	 1.3346607e-01	 1.4535609e+00	 1.6081664e-01	  1.0798180e+00 	 2.1713758e-01


.. parsed-literal::

     102	 1.4044513e+00	 1.3336433e-01	 1.4570812e+00	 1.6075106e-01	  1.0858367e+00 	 2.2893214e-01


.. parsed-literal::

     103	 1.4081451e+00	 1.3311802e-01	 1.4609242e+00	 1.6041319e-01	  1.0884589e+00 	 2.1817207e-01


.. parsed-literal::

     104	 1.4118763e+00	 1.3313948e-01	 1.4647041e+00	 1.6040697e-01	  1.0924069e+00 	 2.2348690e-01


.. parsed-literal::

     105	 1.4143718e+00	 1.3305904e-01	 1.4671614e+00	 1.6034722e-01	  1.0954343e+00 	 2.0691872e-01


.. parsed-literal::

     106	 1.4168155e+00	 1.3286286e-01	 1.4695460e+00	 1.5986172e-01	[ 1.0990448e+00]	 2.1789384e-01


.. parsed-literal::

     107	 1.4198548e+00	 1.3258236e-01	 1.4725326e+00	 1.5939989e-01	[ 1.1022985e+00]	 2.2782850e-01


.. parsed-literal::

     108	 1.4233694e+00	 1.3217452e-01	 1.4761486e+00	 1.5876076e-01	[ 1.1047729e+00]	 2.1958494e-01
     109	 1.4270688e+00	 1.3202463e-01	 1.4798039e+00	 1.5879635e-01	[ 1.1073483e+00]	 1.9600391e-01


.. parsed-literal::

     110	 1.4297566e+00	 1.3187794e-01	 1.4825962e+00	 1.5883284e-01	  1.1048215e+00 	 2.1761131e-01


.. parsed-literal::

     111	 1.4330000e+00	 1.3172244e-01	 1.4859416e+00	 1.5892830e-01	  1.1009645e+00 	 2.2998953e-01


.. parsed-literal::

     112	 1.4362151e+00	 1.3170287e-01	 1.4892560e+00	 1.5929026e-01	  1.0965027e+00 	 2.2490096e-01


.. parsed-literal::

     113	 1.4393920e+00	 1.3155345e-01	 1.4923432e+00	 1.5918998e-01	  1.0947829e+00 	 2.1232891e-01


.. parsed-literal::

     114	 1.4415908e+00	 1.3152086e-01	 1.4944279e+00	 1.5919118e-01	  1.0952596e+00 	 2.2619629e-01


.. parsed-literal::

     115	 1.4440507e+00	 1.3153716e-01	 1.4969003e+00	 1.5919693e-01	  1.0939791e+00 	 2.2171760e-01


.. parsed-literal::

     116	 1.4469312e+00	 1.3151874e-01	 1.5000190e+00	 1.5919089e-01	  1.0858491e+00 	 2.2950864e-01


.. parsed-literal::

     117	 1.4500850e+00	 1.3157930e-01	 1.5032233e+00	 1.5934701e-01	  1.0849061e+00 	 2.2489953e-01


.. parsed-literal::

     118	 1.4519859e+00	 1.3150628e-01	 1.5052182e+00	 1.5921415e-01	  1.0853270e+00 	 2.1499395e-01


.. parsed-literal::

     119	 1.4543723e+00	 1.3137467e-01	 1.5077806e+00	 1.5898388e-01	  1.0832854e+00 	 2.1516228e-01


.. parsed-literal::

     120	 1.4559162e+00	 1.3119686e-01	 1.5096074e+00	 1.5849984e-01	  1.0731779e+00 	 2.1743989e-01


.. parsed-literal::

     121	 1.4581172e+00	 1.3109759e-01	 1.5117345e+00	 1.5833558e-01	  1.0756785e+00 	 2.1637130e-01


.. parsed-literal::

     122	 1.4597045e+00	 1.3103229e-01	 1.5132586e+00	 1.5822752e-01	  1.0757010e+00 	 2.2211957e-01


.. parsed-literal::

     123	 1.4618091e+00	 1.3094621e-01	 1.5153347e+00	 1.5800788e-01	  1.0750944e+00 	 2.1382642e-01


.. parsed-literal::

     124	 1.4634830e+00	 1.3089864e-01	 1.5170824e+00	 1.5768238e-01	  1.0693358e+00 	 2.1377063e-01


.. parsed-literal::

     125	 1.4666655e+00	 1.3073474e-01	 1.5202623e+00	 1.5741396e-01	  1.0690869e+00 	 2.3569489e-01


.. parsed-literal::

     126	 1.4681302e+00	 1.3064096e-01	 1.5217689e+00	 1.5731140e-01	  1.0704635e+00 	 2.2087145e-01
     127	 1.4702800e+00	 1.3045912e-01	 1.5240585e+00	 1.5709854e-01	  1.0682472e+00 	 1.8667841e-01


.. parsed-literal::

     128	 1.4728953e+00	 1.3024026e-01	 1.5268018e+00	 1.5684731e-01	  1.0678447e+00 	 1.8596601e-01


.. parsed-literal::

     129	 1.4758011e+00	 1.2994570e-01	 1.5298610e+00	 1.5644540e-01	  1.0573738e+00 	 2.1903706e-01


.. parsed-literal::

     130	 1.4777899e+00	 1.2983730e-01	 1.5318170e+00	 1.5622236e-01	  1.0546106e+00 	 2.3397708e-01
     131	 1.4802987e+00	 1.2974192e-01	 1.5342156e+00	 1.5605695e-01	  1.0557236e+00 	 1.9798493e-01


.. parsed-literal::

     132	 1.4824936e+00	 1.2959418e-01	 1.5363472e+00	 1.5552601e-01	  1.0495634e+00 	 2.1978831e-01


.. parsed-literal::

     133	 1.4844294e+00	 1.2951284e-01	 1.5382556e+00	 1.5531635e-01	  1.0517828e+00 	 2.1426463e-01


.. parsed-literal::

     134	 1.4866326e+00	 1.2936800e-01	 1.5405462e+00	 1.5489268e-01	  1.0460873e+00 	 2.1921396e-01


.. parsed-literal::

     135	 1.4879805e+00	 1.2935369e-01	 1.5419807e+00	 1.5483234e-01	  1.0456831e+00 	 2.1940255e-01


.. parsed-literal::

     136	 1.4890839e+00	 1.2930853e-01	 1.5431158e+00	 1.5480290e-01	  1.0419311e+00 	 2.2547174e-01


.. parsed-literal::

     137	 1.4927144e+00	 1.2905510e-01	 1.5469389e+00	 1.5449840e-01	  1.0231109e+00 	 2.2224736e-01


.. parsed-literal::

     138	 1.4940172e+00	 1.2878026e-01	 1.5482974e+00	 1.5413932e-01	  1.0121177e+00 	 2.1260166e-01


.. parsed-literal::

     139	 1.4961179e+00	 1.2864951e-01	 1.5503571e+00	 1.5390754e-01	  1.0118748e+00 	 2.0325255e-01


.. parsed-literal::

     140	 1.4980170e+00	 1.2848948e-01	 1.5521570e+00	 1.5368183e-01	  1.0187950e+00 	 2.1368742e-01


.. parsed-literal::

     141	 1.4995352e+00	 1.2829618e-01	 1.5536190e+00	 1.5337560e-01	  1.0206279e+00 	 2.0480013e-01


.. parsed-literal::

     142	 1.5012858e+00	 1.2800989e-01	 1.5553681e+00	 1.5304800e-01	  1.0311388e+00 	 2.1660995e-01


.. parsed-literal::

     143	 1.5031275e+00	 1.2786312e-01	 1.5572328e+00	 1.5279864e-01	  1.0244475e+00 	 2.2510028e-01


.. parsed-literal::

     144	 1.5047260e+00	 1.2767219e-01	 1.5589651e+00	 1.5252271e-01	  1.0115909e+00 	 2.2217417e-01


.. parsed-literal::

     145	 1.5061720e+00	 1.2749559e-01	 1.5605084e+00	 1.5235368e-01	  1.0001250e+00 	 2.1583891e-01


.. parsed-literal::

     146	 1.5083664e+00	 1.2692608e-01	 1.5628532e+00	 1.5171232e-01	  9.6895256e-01 	 2.2495985e-01


.. parsed-literal::

     147	 1.5103581e+00	 1.2668695e-01	 1.5648739e+00	 1.5172237e-01	  9.5765919e-01 	 2.2536540e-01


.. parsed-literal::

     148	 1.5115182e+00	 1.2664208e-01	 1.5659196e+00	 1.5169897e-01	  9.6396284e-01 	 2.1843529e-01


.. parsed-literal::

     149	 1.5133370e+00	 1.2641006e-01	 1.5675963e+00	 1.5148195e-01	  9.6768571e-01 	 2.2915435e-01


.. parsed-literal::

     150	 1.5143528e+00	 1.2592012e-01	 1.5686316e+00	 1.5100682e-01	  9.6353605e-01 	 2.2744727e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 10s, sys: 1.35 s, total: 2min 11s
    Wall time: 33.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f798435ada0>



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
    CPU times: user 1.97 s, sys: 57.9 ms, total: 2.03 s
    Wall time: 733 ms


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

