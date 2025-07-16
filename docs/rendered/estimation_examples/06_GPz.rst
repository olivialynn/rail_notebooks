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
       1	-3.5287964e-01	 3.2341298e-01	-3.4319631e-01	 3.0977703e-01	[-3.1715322e-01]	 4.5727921e-01


.. parsed-literal::

       2	-2.8226103e-01	 3.1303556e-01	-2.5848990e-01	 2.9766479e-01	[-2.1295917e-01]	 2.3059773e-01


.. parsed-literal::

       3	-2.3490526e-01	 2.9027217e-01	-1.9083222e-01	 2.7761450e-01	[-1.3596913e-01]	 2.9299808e-01


.. parsed-literal::

       4	-2.0981037e-01	 2.6802659e-01	-1.6956312e-01	 2.6025077e-01	[-1.1520863e-01]	 2.0670938e-01


.. parsed-literal::

       5	-1.1553575e-01	 2.6077654e-01	-8.1470663e-02	 2.4896118e-01	[-2.8282883e-02]	 2.1694660e-01


.. parsed-literal::

       6	-8.3279675e-02	 2.5456735e-01	-5.2275282e-02	 2.4351933e-01	[-1.2558877e-02]	 2.1694136e-01


.. parsed-literal::

       7	-6.4435106e-02	 2.5179416e-01	-4.0320876e-02	 2.4007515e-01	[ 4.0243143e-03]	 2.0856357e-01


.. parsed-literal::

       8	-5.0339664e-02	 2.4934336e-01	-3.0095540e-02	 2.3727669e-01	[ 1.6724447e-02]	 2.1843719e-01
       9	-3.3940704e-02	 2.4615468e-01	-1.6653266e-02	 2.3433878e-01	[ 2.9741225e-02]	 1.8910003e-01


.. parsed-literal::

      10	-2.3476259e-02	 2.4416918e-01	-8.1388697e-03	 2.3431147e-01	[ 3.2460792e-02]	 2.1055770e-01


.. parsed-literal::

      11	-1.8472143e-02	 2.4336898e-01	-4.0969632e-03	 2.3365824e-01	[ 3.5100909e-02]	 2.0166731e-01


.. parsed-literal::

      12	-1.5766902e-02	 2.4282000e-01	-1.6668705e-03	 2.3303646e-01	[ 3.6651623e-02]	 2.1480441e-01


.. parsed-literal::

      13	-1.1894470e-02	 2.4212154e-01	 2.0924105e-03	 2.3232817e-01	[ 4.1329083e-02]	 2.1958733e-01


.. parsed-literal::

      14	 3.1124795e-02	 2.3177088e-01	 4.7490493e-02	 2.2286732e-01	[ 7.3691981e-02]	 3.2495880e-01


.. parsed-literal::

      15	 4.5932519e-02	 2.2661354e-01	 6.4549879e-02	 2.1731780e-01	[ 8.4028028e-02]	 2.1484494e-01


.. parsed-literal::

      16	 1.5481260e-01	 2.2125740e-01	 1.7621731e-01	 2.1529333e-01	[ 1.9744622e-01]	 2.0425844e-01
      17	 2.4603999e-01	 2.1599412e-01	 2.7692213e-01	 2.0966278e-01	[ 3.1034268e-01]	 1.9311666e-01


.. parsed-literal::

      18	 2.9920490e-01	 2.1724961e-01	 3.2992693e-01	 2.1143522e-01	[ 3.5099787e-01]	 2.0238662e-01
      19	 3.3581377e-01	 2.1278643e-01	 3.6691795e-01	 2.1022266e-01	[ 3.7799800e-01]	 1.7626691e-01


.. parsed-literal::

      20	 3.7707027e-01	 2.0960363e-01	 4.0785242e-01	 2.0816943e-01	[ 4.1802167e-01]	 1.8304491e-01


.. parsed-literal::

      21	 4.3393307e-01	 2.0703214e-01	 4.6586453e-01	 2.0769162e-01	[ 4.7516858e-01]	 2.1265960e-01
      22	 5.2660869e-01	 2.0486500e-01	 5.6151887e-01	 2.0100923e-01	[ 5.7365576e-01]	 1.7725730e-01


.. parsed-literal::

      23	 5.8482074e-01	 2.0233296e-01	 6.2315569e-01	 1.9777712e-01	[ 6.1952811e-01]	 1.9536662e-01


.. parsed-literal::

      24	 6.2474166e-01	 1.9942889e-01	 6.6313504e-01	 1.9323599e-01	[ 6.6058246e-01]	 2.1157813e-01


.. parsed-literal::

      25	 6.5373743e-01	 1.9545606e-01	 6.9128675e-01	 1.9000118e-01	[ 6.8559458e-01]	 2.2080088e-01


.. parsed-literal::

      26	 6.9824388e-01	 1.9118107e-01	 7.3563721e-01	 1.8671234e-01	[ 7.2466006e-01]	 2.1289015e-01
      27	 7.2483259e-01	 1.9751424e-01	 7.6212262e-01	 1.9145926e-01	[ 7.4118613e-01]	 2.0100188e-01


.. parsed-literal::

      28	 7.5064279e-01	 1.9376028e-01	 7.8724746e-01	 1.8987389e-01	[ 7.7501043e-01]	 2.0337200e-01


.. parsed-literal::

      29	 7.6696715e-01	 1.9187684e-01	 8.0436412e-01	 1.8885566e-01	[ 7.9043570e-01]	 2.0387268e-01
      30	 7.9180239e-01	 1.9234511e-01	 8.3061534e-01	 1.8667789e-01	[ 8.2189182e-01]	 1.9621468e-01


.. parsed-literal::

      31	 8.1595358e-01	 1.9118823e-01	 8.5422577e-01	 1.8613147e-01	[ 8.4741369e-01]	 1.9890404e-01


.. parsed-literal::

      32	 8.3489652e-01	 1.9032275e-01	 8.7340313e-01	 1.8641323e-01	[ 8.6584628e-01]	 2.0360804e-01


.. parsed-literal::

      33	 8.6525677e-01	 1.8829720e-01	 9.0415972e-01	 1.8407393e-01	[ 8.9894484e-01]	 2.1498799e-01
      34	 8.9188409e-01	 1.8778872e-01	 9.3195676e-01	 1.8270998e-01	[ 9.3712306e-01]	 1.9997478e-01


.. parsed-literal::

      35	 9.0611860e-01	 1.8655207e-01	 9.4816103e-01	 1.7881211e-01	[ 9.6152974e-01]	 2.0260048e-01


.. parsed-literal::

      36	 9.2812230e-01	 1.8372178e-01	 9.6948823e-01	 1.7618027e-01	[ 9.7746169e-01]	 2.0516920e-01


.. parsed-literal::

      37	 9.4285353e-01	 1.8199189e-01	 9.8451852e-01	 1.7417750e-01	[ 9.8724393e-01]	 2.1107554e-01


.. parsed-literal::

      38	 9.6743134e-01	 1.7941148e-01	 1.0102249e+00	 1.7094251e-01	[ 1.0014602e+00]	 2.0849848e-01
      39	 9.7644565e-01	 1.7796444e-01	 1.0205847e+00	 1.7095038e-01	[ 1.0034104e+00]	 2.0108986e-01


.. parsed-literal::

      40	 9.9551865e-01	 1.7704431e-01	 1.0397666e+00	 1.6998256e-01	[ 1.0248290e+00]	 2.0785189e-01


.. parsed-literal::

      41	 1.0018523e+00	 1.7630279e-01	 1.0459132e+00	 1.6952283e-01	[ 1.0297522e+00]	 2.1118307e-01


.. parsed-literal::

      42	 1.0112968e+00	 1.7547516e-01	 1.0555219e+00	 1.6893444e-01	[ 1.0374133e+00]	 2.0323372e-01
      43	 1.0252953e+00	 1.7516249e-01	 1.0705741e+00	 1.6902531e-01	[ 1.0470288e+00]	 2.0109057e-01


.. parsed-literal::

      44	 1.0382641e+00	 1.7361382e-01	 1.0838674e+00	 1.6798904e-01	[ 1.0575755e+00]	 2.0706487e-01


.. parsed-literal::

      45	 1.0468094e+00	 1.7227586e-01	 1.0925429e+00	 1.6730652e-01	[ 1.0657523e+00]	 2.0852804e-01
      46	 1.0595160e+00	 1.7008998e-01	 1.1057634e+00	 1.6590672e-01	[ 1.0744027e+00]	 1.7449355e-01


.. parsed-literal::

      47	 1.0751288e+00	 1.6643998e-01	 1.1217072e+00	 1.6289935e-01	[ 1.0844167e+00]	 2.0732355e-01


.. parsed-literal::

      48	 1.0862661e+00	 1.6309221e-01	 1.1332419e+00	 1.6078927e-01	[ 1.0853716e+00]	 2.0698595e-01


.. parsed-literal::

      49	 1.0963867e+00	 1.6272088e-01	 1.1430166e+00	 1.5879452e-01	[ 1.1038755e+00]	 2.0915031e-01
      50	 1.1058026e+00	 1.6135064e-01	 1.1528257e+00	 1.5754184e-01	[ 1.1069942e+00]	 1.6995740e-01


.. parsed-literal::

      51	 1.1161997e+00	 1.5940265e-01	 1.1636223e+00	 1.5590233e-01	[ 1.1100470e+00]	 1.9663692e-01


.. parsed-literal::

      52	 1.1307837e+00	 1.5618631e-01	 1.1783819e+00	 1.5297568e-01	[ 1.1169703e+00]	 2.0877266e-01


.. parsed-literal::

      53	 1.1377766e+00	 1.5287464e-01	 1.1857442e+00	 1.5134071e-01	  1.1154065e+00 	 2.0091486e-01
      54	 1.1506875e+00	 1.5239932e-01	 1.1981727e+00	 1.4945445e-01	[ 1.1332095e+00]	 1.8886089e-01


.. parsed-literal::

      55	 1.1557311e+00	 1.5193188e-01	 1.2032148e+00	 1.4915322e-01	[ 1.1347095e+00]	 2.0650005e-01


.. parsed-literal::

      56	 1.1641507e+00	 1.5148191e-01	 1.2118894e+00	 1.4818485e-01	[ 1.1400025e+00]	 2.1055269e-01
      57	 1.1730134e+00	 1.4989444e-01	 1.2213573e+00	 1.4825790e-01	  1.1365943e+00 	 1.9918799e-01


.. parsed-literal::

      58	 1.1804037e+00	 1.4910186e-01	 1.2289125e+00	 1.4829526e-01	  1.1367793e+00 	 2.0580864e-01


.. parsed-literal::

      59	 1.1900586e+00	 1.4798500e-01	 1.2389123e+00	 1.4830129e-01	  1.1391530e+00 	 2.0605302e-01
      60	 1.1990561e+00	 1.4699283e-01	 1.2483081e+00	 1.4802786e-01	  1.1393102e+00 	 1.9921684e-01


.. parsed-literal::

      61	 1.2092307e+00	 1.4650896e-01	 1.2587947e+00	 1.4791710e-01	[ 1.1467966e+00]	 1.9700384e-01


.. parsed-literal::

      62	 1.2163856e+00	 1.4628378e-01	 1.2659119e+00	 1.4720910e-01	[ 1.1517766e+00]	 2.0508862e-01


.. parsed-literal::

      63	 1.2238008e+00	 1.4573739e-01	 1.2734754e+00	 1.4657571e-01	[ 1.1522768e+00]	 2.1352029e-01
      64	 1.2323250e+00	 1.4594780e-01	 1.2821502e+00	 1.4588541e-01	[ 1.1615169e+00]	 2.0000291e-01


.. parsed-literal::

      65	 1.2423190e+00	 1.4553524e-01	 1.2923048e+00	 1.4575200e-01	[ 1.1655895e+00]	 2.1513319e-01


.. parsed-literal::

      66	 1.2525188e+00	 1.4446486e-01	 1.3028647e+00	 1.4532441e-01	[ 1.1733671e+00]	 2.1707082e-01
      67	 1.2602018e+00	 1.4403404e-01	 1.3108470e+00	 1.4485781e-01	[ 1.1828224e+00]	 2.0002484e-01


.. parsed-literal::

      68	 1.2677467e+00	 1.4290931e-01	 1.3184799e+00	 1.4388358e-01	[ 1.1872331e+00]	 1.7903399e-01


.. parsed-literal::

      69	 1.2752174e+00	 1.4161808e-01	 1.3261250e+00	 1.4250029e-01	[ 1.1926763e+00]	 2.1152854e-01


.. parsed-literal::

      70	 1.2832775e+00	 1.4061294e-01	 1.3343822e+00	 1.4116375e-01	[ 1.1954070e+00]	 2.0847797e-01


.. parsed-literal::

      71	 1.2907822e+00	 1.3944084e-01	 1.3421171e+00	 1.4022765e-01	[ 1.2038636e+00]	 2.0245242e-01


.. parsed-literal::

      72	 1.2968096e+00	 1.3945012e-01	 1.3482251e+00	 1.3973716e-01	[ 1.2141965e+00]	 2.0433617e-01
      73	 1.3023450e+00	 1.3911343e-01	 1.3539170e+00	 1.3968221e-01	[ 1.2200555e+00]	 1.8583798e-01


.. parsed-literal::

      74	 1.3068816e+00	 1.3939879e-01	 1.3584828e+00	 1.3970142e-01	[ 1.2258278e+00]	 1.9467378e-01


.. parsed-literal::

      75	 1.3119109e+00	 1.3876466e-01	 1.3636459e+00	 1.3914514e-01	[ 1.2291912e+00]	 2.0603204e-01


.. parsed-literal::

      76	 1.3205531e+00	 1.3757993e-01	 1.3725664e+00	 1.3822446e-01	  1.2277210e+00 	 2.1111631e-01


.. parsed-literal::

      77	 1.3246913e+00	 1.3754743e-01	 1.3769120e+00	 1.3747593e-01	  1.2289874e+00 	 2.1688700e-01
      78	 1.3306790e+00	 1.3735642e-01	 1.3827189e+00	 1.3756731e-01	[ 1.2337404e+00]	 1.8433452e-01


.. parsed-literal::

      79	 1.3342373e+00	 1.3733387e-01	 1.3862973e+00	 1.3769542e-01	[ 1.2367818e+00]	 2.0581055e-01


.. parsed-literal::

      80	 1.3383538e+00	 1.3737836e-01	 1.3905086e+00	 1.3751513e-01	[ 1.2415214e+00]	 2.0800304e-01
      81	 1.3423915e+00	 1.3699065e-01	 1.3951060e+00	 1.3671218e-01	[ 1.2429399e+00]	 1.9838881e-01


.. parsed-literal::

      82	 1.3490053e+00	 1.3686297e-01	 1.4016534e+00	 1.3608670e-01	[ 1.2517669e+00]	 2.0489693e-01


.. parsed-literal::

      83	 1.3525359e+00	 1.3648607e-01	 1.4052198e+00	 1.3533563e-01	[ 1.2553283e+00]	 2.0540023e-01
      84	 1.3566572e+00	 1.3592186e-01	 1.4094485e+00	 1.3446220e-01	[ 1.2595197e+00]	 1.9928765e-01


.. parsed-literal::

      85	 1.3604662e+00	 1.3541981e-01	 1.4135030e+00	 1.3354712e-01	[ 1.2660124e+00]	 2.0912886e-01


.. parsed-literal::

      86	 1.3654852e+00	 1.3482975e-01	 1.4184511e+00	 1.3319549e-01	[ 1.2702641e+00]	 2.1465492e-01
      87	 1.3689506e+00	 1.3445893e-01	 1.4218388e+00	 1.3321173e-01	[ 1.2743427e+00]	 2.0000744e-01


.. parsed-literal::

      88	 1.3723755e+00	 1.3413826e-01	 1.4252953e+00	 1.3303249e-01	[ 1.2785847e+00]	 1.8165731e-01
      89	 1.3760468e+00	 1.3355223e-01	 1.4290388e+00	 1.3224708e-01	[ 1.2865676e+00]	 1.8967032e-01


.. parsed-literal::

      90	 1.3807350e+00	 1.3321835e-01	 1.4337028e+00	 1.3182919e-01	[ 1.2911850e+00]	 2.0917511e-01


.. parsed-literal::

      91	 1.3843585e+00	 1.3292308e-01	 1.4372678e+00	 1.3140193e-01	[ 1.2934134e+00]	 2.0933914e-01
      92	 1.3880812e+00	 1.3252710e-01	 1.4410151e+00	 1.3082684e-01	[ 1.2957361e+00]	 1.9882441e-01


.. parsed-literal::

      93	 1.3922113e+00	 1.3191503e-01	 1.4451644e+00	 1.3067161e-01	  1.2949306e+00 	 2.1221995e-01


.. parsed-literal::

      94	 1.3971420e+00	 1.3158992e-01	 1.4500963e+00	 1.3002295e-01	[ 1.3026617e+00]	 2.0889711e-01
      95	 1.4006047e+00	 1.3139320e-01	 1.4534987e+00	 1.2985038e-01	[ 1.3073125e+00]	 1.8782234e-01


.. parsed-literal::

      96	 1.4056306e+00	 1.3119744e-01	 1.4586234e+00	 1.2958871e-01	[ 1.3154577e+00]	 1.8742323e-01


.. parsed-literal::

      97	 1.4086770e+00	 1.3096157e-01	 1.4618286e+00	 1.2904210e-01	[ 1.3200121e+00]	 2.1680593e-01


.. parsed-literal::

      98	 1.4126674e+00	 1.3084860e-01	 1.4657482e+00	 1.2880055e-01	[ 1.3238068e+00]	 2.0087314e-01


.. parsed-literal::

      99	 1.4166299e+00	 1.3064569e-01	 1.4697587e+00	 1.2851144e-01	  1.3234340e+00 	 2.1388960e-01
     100	 1.4188353e+00	 1.3057808e-01	 1.4720144e+00	 1.2833389e-01	[ 1.3244783e+00]	 1.8633604e-01


.. parsed-literal::

     101	 1.4211647e+00	 1.3049747e-01	 1.4743601e+00	 1.2835476e-01	  1.3235572e+00 	 2.0923376e-01
     102	 1.4256849e+00	 1.3037275e-01	 1.4790960e+00	 1.2839506e-01	  1.3180371e+00 	 1.8846846e-01


.. parsed-literal::

     103	 1.4274253e+00	 1.3019266e-01	 1.4809805e+00	 1.2862723e-01	  1.3134449e+00 	 2.0468998e-01


.. parsed-literal::

     104	 1.4301667e+00	 1.3014551e-01	 1.4836831e+00	 1.2830219e-01	  1.3171523e+00 	 2.1528816e-01
     105	 1.4327777e+00	 1.3001562e-01	 1.4863358e+00	 1.2782878e-01	  1.3207978e+00 	 2.0148277e-01


.. parsed-literal::

     106	 1.4348993e+00	 1.2986078e-01	 1.4884584e+00	 1.2746407e-01	  1.3234483e+00 	 2.0893717e-01


.. parsed-literal::

     107	 1.4395782e+00	 1.2953839e-01	 1.4932234e+00	 1.2705332e-01	[ 1.3248114e+00]	 2.0613480e-01
     108	 1.4412421e+00	 1.2911950e-01	 1.4951147e+00	 1.2563926e-01	  1.3202921e+00 	 1.7226267e-01


.. parsed-literal::

     109	 1.4472498e+00	 1.2912553e-01	 1.5009449e+00	 1.2590982e-01	[ 1.3251853e+00]	 2.0334196e-01
     110	 1.4496628e+00	 1.2907311e-01	 1.5033419e+00	 1.2609278e-01	[ 1.3253510e+00]	 1.7557621e-01


.. parsed-literal::

     111	 1.4525041e+00	 1.2900677e-01	 1.5062454e+00	 1.2601561e-01	[ 1.3266332e+00]	 2.1253943e-01
     112	 1.4551540e+00	 1.2903258e-01	 1.5090620e+00	 1.2564559e-01	  1.3248187e+00 	 1.8180060e-01


.. parsed-literal::

     113	 1.4580061e+00	 1.2889136e-01	 1.5118779e+00	 1.2540072e-01	[ 1.3297659e+00]	 1.6954780e-01


.. parsed-literal::

     114	 1.4607885e+00	 1.2867292e-01	 1.5146343e+00	 1.2489609e-01	[ 1.3366385e+00]	 2.1782112e-01


.. parsed-literal::

     115	 1.4626475e+00	 1.2841597e-01	 1.5165357e+00	 1.2457530e-01	[ 1.3366833e+00]	 2.0503283e-01
     116	 1.4648850e+00	 1.2829402e-01	 1.5187879e+00	 1.2429135e-01	[ 1.3402276e+00]	 1.9883752e-01


.. parsed-literal::

     117	 1.4676637e+00	 1.2808009e-01	 1.5215921e+00	 1.2405658e-01	  1.3402048e+00 	 2.0828342e-01
     118	 1.4707207e+00	 1.2779577e-01	 1.5247112e+00	 1.2389228e-01	  1.3369647e+00 	 1.8704200e-01


.. parsed-literal::

     119	 1.4733581e+00	 1.2746059e-01	 1.5274347e+00	 1.2395463e-01	  1.3305300e+00 	 2.0717025e-01
     120	 1.4755728e+00	 1.2726347e-01	 1.5295978e+00	 1.2392471e-01	  1.3303579e+00 	 1.7334175e-01


.. parsed-literal::

     121	 1.4777892e+00	 1.2704255e-01	 1.5318340e+00	 1.2395893e-01	  1.3292776e+00 	 2.0742106e-01


.. parsed-literal::

     122	 1.4797124e+00	 1.2682456e-01	 1.5338093e+00	 1.2402135e-01	  1.3285236e+00 	 2.1600008e-01


.. parsed-literal::

     123	 1.4832552e+00	 1.2648902e-01	 1.5375749e+00	 1.2431513e-01	  1.3221867e+00 	 2.0639253e-01
     124	 1.4860371e+00	 1.2634139e-01	 1.5405346e+00	 1.2494249e-01	  1.3201018e+00 	 1.8014002e-01


.. parsed-literal::

     125	 1.4881860e+00	 1.2625563e-01	 1.5426171e+00	 1.2469076e-01	  1.3223896e+00 	 2.0971608e-01
     126	 1.4902815e+00	 1.2614182e-01	 1.5447650e+00	 1.2460500e-01	  1.3196392e+00 	 1.9276237e-01


.. parsed-literal::

     127	 1.4918271e+00	 1.2611876e-01	 1.5464276e+00	 1.2449281e-01	  1.3207639e+00 	 1.9337821e-01
     128	 1.4938201e+00	 1.2594378e-01	 1.5484970e+00	 1.2439177e-01	  1.3187411e+00 	 1.8393230e-01


.. parsed-literal::

     129	 1.4954965e+00	 1.2584673e-01	 1.5502263e+00	 1.2436366e-01	  1.3173956e+00 	 2.0229292e-01


.. parsed-literal::

     130	 1.4968906e+00	 1.2574972e-01	 1.5516241e+00	 1.2426691e-01	  1.3187888e+00 	 2.0734334e-01


.. parsed-literal::

     131	 1.4986848e+00	 1.2567415e-01	 1.5534780e+00	 1.2418095e-01	  1.3157419e+00 	 2.2632504e-01


.. parsed-literal::

     132	 1.5001764e+00	 1.2561419e-01	 1.5549383e+00	 1.2417665e-01	  1.3170727e+00 	 2.1849012e-01
     133	 1.5013802e+00	 1.2558673e-01	 1.5561505e+00	 1.2412131e-01	  1.3169693e+00 	 1.8624377e-01


.. parsed-literal::

     134	 1.5031475e+00	 1.2550532e-01	 1.5579764e+00	 1.2398364e-01	  1.3156242e+00 	 2.1431088e-01


.. parsed-literal::

     135	 1.5043049e+00	 1.2539454e-01	 1.5593288e+00	 1.2365762e-01	  1.3097655e+00 	 2.1737242e-01
     136	 1.5060691e+00	 1.2532554e-01	 1.5610458e+00	 1.2355880e-01	  1.3116735e+00 	 1.9946241e-01


.. parsed-literal::

     137	 1.5073853e+00	 1.2524206e-01	 1.5623560e+00	 1.2344514e-01	  1.3119324e+00 	 1.9895411e-01
     138	 1.5087836e+00	 1.2515386e-01	 1.5637498e+00	 1.2335382e-01	  1.3115246e+00 	 1.7872548e-01


.. parsed-literal::

     139	 1.5110587e+00	 1.2510096e-01	 1.5660711e+00	 1.2316058e-01	  1.3092080e+00 	 2.1085739e-01


.. parsed-literal::

     140	 1.5123363e+00	 1.2496213e-01	 1.5673604e+00	 1.2319364e-01	  1.3062974e+00 	 3.3139920e-01


.. parsed-literal::

     141	 1.5136340e+00	 1.2497823e-01	 1.5686453e+00	 1.2318906e-01	  1.3067684e+00 	 2.0668054e-01
     142	 1.5150402e+00	 1.2493676e-01	 1.5700776e+00	 1.2305636e-01	  1.3064304e+00 	 1.8421483e-01


.. parsed-literal::

     143	 1.5168068e+00	 1.2481170e-01	 1.5718922e+00	 1.2278027e-01	  1.3059343e+00 	 2.0486856e-01


.. parsed-literal::

     144	 1.5182805e+00	 1.2451627e-01	 1.5735460e+00	 1.2222421e-01	  1.3058645e+00 	 2.0564055e-01


.. parsed-literal::

     145	 1.5203455e+00	 1.2443165e-01	 1.5755904e+00	 1.2203546e-01	  1.3047470e+00 	 2.1385431e-01


.. parsed-literal::

     146	 1.5211907e+00	 1.2440268e-01	 1.5764099e+00	 1.2201729e-01	  1.3060202e+00 	 2.0431828e-01


.. parsed-literal::

     147	 1.5224526e+00	 1.2433134e-01	 1.5776804e+00	 1.2194120e-01	  1.3060695e+00 	 2.0238471e-01


.. parsed-literal::

     148	 1.5237066e+00	 1.2437874e-01	 1.5789903e+00	 1.2195801e-01	  1.3056757e+00 	 2.0197797e-01
     149	 1.5251107e+00	 1.2432246e-01	 1.5804014e+00	 1.2190146e-01	  1.3045374e+00 	 1.8599725e-01


.. parsed-literal::

     150	 1.5263205e+00	 1.2426854e-01	 1.5816349e+00	 1.2182593e-01	  1.3035273e+00 	 2.1833634e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.01 s, total: 2min 4s
    Wall time: 31.2 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f0cbc469060>



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
    CPU times: user 2.01 s, sys: 46.9 ms, total: 2.05 s
    Wall time: 610 ms


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

