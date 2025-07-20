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
       1	-3.3931973e-01	 3.1927024e-01	-3.2960350e-01	 3.2536101e-01	[-3.4167101e-01]	 4.5281911e-01


.. parsed-literal::

       2	-2.6814773e-01	 3.0841261e-01	-2.4388914e-01	 3.1531861e-01	[-2.6514460e-01]	 2.2702193e-01


.. parsed-literal::

       3	-2.2547545e-01	 2.8878686e-01	-1.8388170e-01	 2.9451580e-01	[-2.1030773e-01]	 2.8464222e-01
       4	-1.8619360e-01	 2.6437579e-01	-1.4347495e-01	 2.7209408e-01	[-1.8913935e-01]	 1.9718671e-01


.. parsed-literal::

       5	-1.0127315e-01	 2.5552095e-01	-6.6859616e-02	 2.6417427e-01	[-1.1226707e-01]	 2.0738292e-01
       6	-6.4179586e-02	 2.4983211e-01	-3.3303243e-02	 2.5893168e-01	[-6.6096115e-02]	 1.7760181e-01


.. parsed-literal::

       7	-4.7758541e-02	 2.4742245e-01	-2.2738844e-02	 2.5646664e-01	[-5.8350462e-02]	 2.1321845e-01
       8	-3.2629809e-02	 2.4489099e-01	-1.2057646e-02	 2.5363885e-01	[-4.7825467e-02]	 1.9162345e-01


.. parsed-literal::

       9	-1.9078934e-02	 2.4243588e-01	-1.5531552e-03	 2.5082421e-01	[-3.6972112e-02]	 1.6970587e-01


.. parsed-literal::

      10	-9.8591783e-03	 2.4105232e-01	 5.2370175e-03	 2.5044739e-01	[-3.3769658e-02]	 2.1864009e-01
      11	-3.1938823e-03	 2.3961854e-01	 1.1122186e-02	 2.4938092e-01	[-3.0443157e-02]	 1.8025923e-01


.. parsed-literal::

      12	-1.0746037e-03	 2.3922058e-01	 1.3074228e-02	 2.4886310e-01	[-2.7387898e-02]	 2.0188975e-01


.. parsed-literal::

      13	 4.5131978e-03	 2.3813792e-01	 1.8729396e-02	 2.4767428e-01	[-2.3310019e-02]	 2.0641208e-01


.. parsed-literal::

      14	 2.6822864e-02	 2.2752524e-01	 4.2139776e-02	 2.3701414e-01	[ 1.6336082e-02]	 2.8726935e-01
      15	 7.7159212e-02	 2.2352637e-01	 9.4830149e-02	 2.3275154e-01	[ 6.5682471e-02]	 1.9029307e-01


.. parsed-literal::

      16	 1.5049982e-01	 2.2189253e-01	 1.7939422e-01	 2.2816467e-01	[ 1.3583380e-01]	 2.0638347e-01
      17	 2.3569463e-01	 2.2338694e-01	 2.6428188e-01	 2.3341041e-01	[ 2.1830797e-01]	 1.9619107e-01


.. parsed-literal::

      18	 2.6479611e-01	 2.1929092e-01	 2.9292631e-01	 2.3006392e-01	[ 2.4678932e-01]	 1.9100547e-01
      19	 3.0105625e-01	 2.1565099e-01	 3.3067112e-01	 2.2756446e-01	[ 2.7778277e-01]	 1.9988823e-01


.. parsed-literal::

      20	 3.7944129e-01	 2.1090366e-01	 4.1209829e-01	 2.2086349e-01	[ 3.5117736e-01]	 2.1524286e-01


.. parsed-literal::

      21	 4.9453663e-01	 2.0818339e-01	 5.3043603e-01	 2.1791695e-01	[ 4.6031877e-01]	 2.0618653e-01
      22	 5.4313641e-01	 2.0779414e-01	 5.7910541e-01	 2.1396197e-01	[ 5.1927029e-01]	 1.9099927e-01


.. parsed-literal::

      23	 5.9173685e-01	 2.0252534e-01	 6.2862801e-01	 2.0870711e-01	[ 5.5423239e-01]	 1.9860601e-01
      24	 6.2839286e-01	 1.9848690e-01	 6.6687957e-01	 2.0590567e-01	[ 5.7451435e-01]	 1.9936681e-01


.. parsed-literal::

      25	 6.6231197e-01	 1.9724431e-01	 7.0045601e-01	 2.0316157e-01	[ 6.2478511e-01]	 1.8245792e-01


.. parsed-literal::

      26	 6.9061797e-01	 1.9544881e-01	 7.2875465e-01	 2.0087272e-01	[ 6.5177294e-01]	 2.0718527e-01
      27	 7.2947275e-01	 1.9762899e-01	 7.6830029e-01	 2.0521320e-01	[ 6.8811572e-01]	 1.9125581e-01


.. parsed-literal::

      28	 7.5713779e-01	 2.0424220e-01	 7.9633520e-01	 2.1374568e-01	[ 7.3339010e-01]	 1.7581034e-01
      29	 7.8500517e-01	 2.0045106e-01	 8.2484699e-01	 2.1101727e-01	[ 7.6243595e-01]	 1.9138908e-01


.. parsed-literal::

      30	 8.0257467e-01	 1.9879232e-01	 8.4318823e-01	 2.0876898e-01	[ 7.7556541e-01]	 2.0989752e-01
      31	 8.3592643e-01	 1.9685316e-01	 8.7809608e-01	 2.0589985e-01	[ 8.0963658e-01]	 2.0049810e-01


.. parsed-literal::

      32	 8.6624790e-01	 1.9571616e-01	 9.0959346e-01	 2.0349818e-01	[ 8.5509469e-01]	 2.0746899e-01


.. parsed-literal::

      33	 8.9456818e-01	 1.9325273e-01	 9.3813215e-01	 2.0062798e-01	[ 8.7933589e-01]	 2.0674944e-01
      34	 9.1283562e-01	 1.9055654e-01	 9.5685271e-01	 1.9700580e-01	[ 8.9179742e-01]	 1.7635608e-01


.. parsed-literal::

      35	 9.4194931e-01	 1.8657354e-01	 9.8700892e-01	 1.9167411e-01	[ 9.0777769e-01]	 1.9891644e-01
      36	 9.6511335e-01	 1.8432781e-01	 1.0109148e+00	 1.8825462e-01	[ 9.2078755e-01]	 1.9904494e-01


.. parsed-literal::

      37	 9.8372933e-01	 1.8054371e-01	 1.0300117e+00	 1.8376501e-01	[ 9.3898535e-01]	 2.1420383e-01


.. parsed-literal::

      38	 9.9944699e-01	 1.7760815e-01	 1.0466339e+00	 1.8063852e-01	[ 9.4515868e-01]	 2.0589709e-01
      39	 1.0225465e+00	 1.7302998e-01	 1.0705439e+00	 1.7686738e-01	[ 9.5424283e-01]	 1.9228911e-01


.. parsed-literal::

      40	 1.0393790e+00	 1.7225001e-01	 1.0884157e+00	 1.7536672e-01	[ 9.6128851e-01]	 1.7711639e-01
      41	 1.0592599e+00	 1.7045806e-01	 1.1082126e+00	 1.7367671e-01	[ 9.8413487e-01]	 2.0010972e-01


.. parsed-literal::

      42	 1.0726579e+00	 1.6945730e-01	 1.1210303e+00	 1.7287851e-01	[ 1.0081292e+00]	 1.7053723e-01


.. parsed-literal::

      43	 1.0876185e+00	 1.6794658e-01	 1.1363349e+00	 1.7132665e-01	[ 1.0243313e+00]	 2.0866680e-01
      44	 1.1054667e+00	 1.6716718e-01	 1.1541968e+00	 1.7051703e-01	[ 1.0391328e+00]	 1.7361593e-01


.. parsed-literal::

      45	 1.1213737e+00	 1.6613601e-01	 1.1703773e+00	 1.6969590e-01	  1.0354633e+00 	 1.7965722e-01
      46	 1.1359221e+00	 1.6377699e-01	 1.1852206e+00	 1.6682521e-01	[ 1.0401229e+00]	 1.7492938e-01


.. parsed-literal::

      47	 1.1496490e+00	 1.6132564e-01	 1.1990200e+00	 1.6420171e-01	[ 1.0418390e+00]	 2.0840001e-01
      48	 1.1628362e+00	 1.5897137e-01	 1.2124992e+00	 1.6156247e-01	[ 1.0441569e+00]	 1.7341876e-01


.. parsed-literal::

      49	 1.1738256e+00	 1.5826189e-01	 1.2233446e+00	 1.6086427e-01	[ 1.0530532e+00]	 2.0523214e-01


.. parsed-literal::

      50	 1.1816249e+00	 1.5827529e-01	 1.2309682e+00	 1.6084088e-01	[ 1.0607888e+00]	 2.1820998e-01
      51	 1.1923965e+00	 1.5819701e-01	 1.2420563e+00	 1.6052807e-01	[ 1.0674344e+00]	 1.7021370e-01


.. parsed-literal::

      52	 1.2034831e+00	 1.5823200e-01	 1.2532646e+00	 1.6027257e-01	[ 1.0839884e+00]	 2.1062589e-01
      53	 1.2154335e+00	 1.5763628e-01	 1.2655878e+00	 1.5950931e-01	[ 1.0995279e+00]	 1.9102597e-01


.. parsed-literal::

      54	 1.2263070e+00	 1.5625401e-01	 1.2764514e+00	 1.5794245e-01	[ 1.1123742e+00]	 2.1219063e-01
      55	 1.2351002e+00	 1.5438897e-01	 1.2851957e+00	 1.5618233e-01	[ 1.1182861e+00]	 1.8102884e-01


.. parsed-literal::

      56	 1.2484494e+00	 1.5205417e-01	 1.2988559e+00	 1.5377907e-01	[ 1.1248411e+00]	 1.9527173e-01


.. parsed-literal::

      57	 1.2590013e+00	 1.5005284e-01	 1.3094196e+00	 1.5165632e-01	[ 1.1326505e+00]	 2.0652151e-01


.. parsed-literal::

      58	 1.2689885e+00	 1.4989339e-01	 1.3193870e+00	 1.5136576e-01	[ 1.1362638e+00]	 2.0447040e-01
      59	 1.2760576e+00	 1.5000908e-01	 1.3266630e+00	 1.5109241e-01	[ 1.1381149e+00]	 1.7529678e-01


.. parsed-literal::

      60	 1.2842801e+00	 1.4946908e-01	 1.3353487e+00	 1.4999139e-01	  1.1352391e+00 	 1.7957401e-01
      61	 1.2909778e+00	 1.4810635e-01	 1.3425423e+00	 1.4807216e-01	  1.1302191e+00 	 1.8408513e-01


.. parsed-literal::

      62	 1.2976362e+00	 1.4731552e-01	 1.3490175e+00	 1.4694945e-01	[ 1.1397857e+00]	 2.0304155e-01


.. parsed-literal::

      63	 1.3039032e+00	 1.4686599e-01	 1.3550452e+00	 1.4634230e-01	[ 1.1504801e+00]	 2.0584154e-01


.. parsed-literal::

      64	 1.3147297e+00	 1.4656207e-01	 1.3659218e+00	 1.4567957e-01	[ 1.1634984e+00]	 2.1059489e-01


.. parsed-literal::

      65	 1.3195127e+00	 1.4691902e-01	 1.3712546e+00	 1.4597772e-01	[ 1.1663365e+00]	 2.0252681e-01


.. parsed-literal::

      66	 1.3299065e+00	 1.4647442e-01	 1.3813509e+00	 1.4551149e-01	[ 1.1790930e+00]	 2.1261239e-01


.. parsed-literal::

      67	 1.3340926e+00	 1.4624055e-01	 1.3856520e+00	 1.4526393e-01	[ 1.1793659e+00]	 2.1453643e-01


.. parsed-literal::

      68	 1.3404317e+00	 1.4557522e-01	 1.3922700e+00	 1.4482861e-01	  1.1792494e+00 	 2.1156359e-01
      69	 1.3464484e+00	 1.4486837e-01	 1.3984300e+00	 1.4434386e-01	[ 1.1844001e+00]	 2.0059562e-01


.. parsed-literal::

      70	 1.3516998e+00	 1.4334606e-01	 1.4038317e+00	 1.4334546e-01	  1.1826498e+00 	 2.1276140e-01
      71	 1.3563023e+00	 1.4279992e-01	 1.4082030e+00	 1.4325790e-01	[ 1.1907416e+00]	 1.8189883e-01


.. parsed-literal::

      72	 1.3614996e+00	 1.4217242e-01	 1.4133878e+00	 1.4286715e-01	[ 1.1991661e+00]	 1.9914317e-01
      73	 1.3674555e+00	 1.4077594e-01	 1.4195276e+00	 1.4218971e-01	[ 1.2023286e+00]	 1.8214488e-01


.. parsed-literal::

      74	 1.3739711e+00	 1.4004173e-01	 1.4263102e+00	 1.4167924e-01	[ 1.2070257e+00]	 1.7403865e-01
      75	 1.3789943e+00	 1.3971631e-01	 1.4313782e+00	 1.4124950e-01	[ 1.2096453e+00]	 2.0492458e-01


.. parsed-literal::

      76	 1.3844961e+00	 1.3944803e-01	 1.4371377e+00	 1.4093526e-01	  1.2053925e+00 	 2.0083094e-01


.. parsed-literal::

      77	 1.3896905e+00	 1.3927990e-01	 1.4423958e+00	 1.4080490e-01	  1.2051987e+00 	 2.0692325e-01
      78	 1.3937240e+00	 1.3902943e-01	 1.4464527e+00	 1.4066233e-01	  1.2063650e+00 	 1.9811273e-01


.. parsed-literal::

      79	 1.4010036e+00	 1.3826855e-01	 1.4539219e+00	 1.4023659e-01	  1.2010152e+00 	 1.9670725e-01
      80	 1.4026711e+00	 1.3769806e-01	 1.4558712e+00	 1.3924227e-01	  1.2077140e+00 	 1.9435620e-01


.. parsed-literal::

      81	 1.4074678e+00	 1.3760690e-01	 1.4604473e+00	 1.3925326e-01	[ 1.2105379e+00]	 2.0810938e-01
      82	 1.4109293e+00	 1.3736945e-01	 1.4639518e+00	 1.3912695e-01	  1.2095820e+00 	 1.9749928e-01


.. parsed-literal::

      83	 1.4150790e+00	 1.3711928e-01	 1.4681496e+00	 1.3866793e-01	[ 1.2133416e+00]	 2.1587944e-01


.. parsed-literal::

      84	 1.4175560e+00	 1.3692605e-01	 1.4707880e+00	 1.3833381e-01	[ 1.2179997e+00]	 2.0779085e-01


.. parsed-literal::

      85	 1.4224024e+00	 1.3673697e-01	 1.4755238e+00	 1.3797891e-01	[ 1.2241556e+00]	 2.0335889e-01
      86	 1.4250342e+00	 1.3659788e-01	 1.4781582e+00	 1.3777333e-01	[ 1.2272851e+00]	 1.8772697e-01


.. parsed-literal::

      87	 1.4277454e+00	 1.3642834e-01	 1.4808790e+00	 1.3762525e-01	[ 1.2316706e+00]	 2.0584202e-01


.. parsed-literal::

      88	 1.4321777e+00	 1.3598567e-01	 1.4853729e+00	 1.3766361e-01	[ 1.2337769e+00]	 2.0917034e-01


.. parsed-literal::

      89	 1.4335534e+00	 1.3588691e-01	 1.4868807e+00	 1.3749874e-01	[ 1.2413137e+00]	 2.1426034e-01


.. parsed-literal::

      90	 1.4380404e+00	 1.3554734e-01	 1.4911988e+00	 1.3756431e-01	  1.2399476e+00 	 2.1807551e-01
      91	 1.4401444e+00	 1.3537038e-01	 1.4932982e+00	 1.3760764e-01	  1.2377475e+00 	 1.9392490e-01


.. parsed-literal::

      92	 1.4432881e+00	 1.3509452e-01	 1.4964797e+00	 1.3762455e-01	  1.2335631e+00 	 1.9910765e-01
      93	 1.4474311e+00	 1.3460113e-01	 1.5007550e+00	 1.3742222e-01	  1.2313000e+00 	 1.6466951e-01


.. parsed-literal::

      94	 1.4515076e+00	 1.3426820e-01	 1.5050111e+00	 1.3764607e-01	  1.2200937e+00 	 1.9855452e-01
      95	 1.4547349e+00	 1.3407163e-01	 1.5082167e+00	 1.3732884e-01	  1.2236266e+00 	 1.8409157e-01


.. parsed-literal::

      96	 1.4584216e+00	 1.3391736e-01	 1.5119375e+00	 1.3707296e-01	  1.2282680e+00 	 2.1439815e-01
      97	 1.4608430e+00	 1.3368828e-01	 1.5144565e+00	 1.3681108e-01	  1.2255869e+00 	 1.9069433e-01


.. parsed-literal::

      98	 1.4635437e+00	 1.3349243e-01	 1.5171867e+00	 1.3666618e-01	  1.2223449e+00 	 2.0862865e-01
      99	 1.4672321e+00	 1.3313199e-01	 1.5209759e+00	 1.3640829e-01	  1.2147184e+00 	 1.7175579e-01


.. parsed-literal::

     100	 1.4693549e+00	 1.3290594e-01	 1.5230985e+00	 1.3621941e-01	  1.2118654e+00 	 2.4462938e-01


.. parsed-literal::

     101	 1.4721489e+00	 1.3266879e-01	 1.5258958e+00	 1.3595670e-01	  1.2138922e+00 	 2.0411181e-01


.. parsed-literal::

     102	 1.4760298e+00	 1.3231389e-01	 1.5298502e+00	 1.3589740e-01	  1.2136764e+00 	 2.1137619e-01


.. parsed-literal::

     103	 1.4770384e+00	 1.3235277e-01	 1.5310608e+00	 1.3530716e-01	  1.2128227e+00 	 2.0449424e-01


.. parsed-literal::

     104	 1.4803543e+00	 1.3224042e-01	 1.5342227e+00	 1.3553888e-01	  1.2188970e+00 	 2.0706296e-01


.. parsed-literal::

     105	 1.4822380e+00	 1.3218354e-01	 1.5361104e+00	 1.3564419e-01	  1.2192468e+00 	 2.0192456e-01


.. parsed-literal::

     106	 1.4845935e+00	 1.3218279e-01	 1.5385120e+00	 1.3568033e-01	  1.2204885e+00 	 2.1263123e-01


.. parsed-literal::

     107	 1.4877713e+00	 1.3209667e-01	 1.5417668e+00	 1.3558697e-01	  1.2233379e+00 	 2.1137142e-01


.. parsed-literal::

     108	 1.4888262e+00	 1.3218222e-01	 1.5430244e+00	 1.3556785e-01	  1.2237410e+00 	 2.2759533e-01


.. parsed-literal::

     109	 1.4925959e+00	 1.3195840e-01	 1.5466323e+00	 1.3535149e-01	  1.2307330e+00 	 2.1750879e-01


.. parsed-literal::

     110	 1.4937564e+00	 1.3184412e-01	 1.5477517e+00	 1.3529922e-01	  1.2332613e+00 	 2.1890759e-01
     111	 1.4964629e+00	 1.3165877e-01	 1.5504498e+00	 1.3528709e-01	  1.2378149e+00 	 1.9843316e-01


.. parsed-literal::

     112	 1.4977563e+00	 1.3144836e-01	 1.5518601e+00	 1.3541487e-01	  1.2405608e+00 	 1.9840670e-01


.. parsed-literal::

     113	 1.5007117e+00	 1.3142862e-01	 1.5547783e+00	 1.3540822e-01	[ 1.2414215e+00]	 2.0891690e-01
     114	 1.5019168e+00	 1.3142918e-01	 1.5559901e+00	 1.3539004e-01	  1.2410758e+00 	 1.9745135e-01


.. parsed-literal::

     115	 1.5035788e+00	 1.3135342e-01	 1.5577132e+00	 1.3529947e-01	  1.2402241e+00 	 2.0338941e-01
     116	 1.5062684e+00	 1.3110388e-01	 1.5604824e+00	 1.3497337e-01	  1.2376201e+00 	 1.7405105e-01


.. parsed-literal::

     117	 1.5073183e+00	 1.3083797e-01	 1.5617961e+00	 1.3447366e-01	  1.2373899e+00 	 1.6608691e-01
     118	 1.5112787e+00	 1.3055321e-01	 1.5656124e+00	 1.3432932e-01	  1.2342739e+00 	 1.8067956e-01


.. parsed-literal::

     119	 1.5124134e+00	 1.3045572e-01	 1.5667166e+00	 1.3427255e-01	  1.2338999e+00 	 2.0914197e-01


.. parsed-literal::

     120	 1.5147845e+00	 1.3023626e-01	 1.5690642e+00	 1.3418669e-01	  1.2312370e+00 	 2.1665812e-01


.. parsed-literal::

     121	 1.5170748e+00	 1.2999973e-01	 1.5714040e+00	 1.3410759e-01	  1.2195024e+00 	 2.1570945e-01
     122	 1.5195261e+00	 1.2984901e-01	 1.5738548e+00	 1.3412965e-01	  1.2181820e+00 	 1.7501259e-01


.. parsed-literal::

     123	 1.5215982e+00	 1.2972841e-01	 1.5759355e+00	 1.3422601e-01	  1.2155836e+00 	 2.0988822e-01


.. parsed-literal::

     124	 1.5225782e+00	 1.2963430e-01	 1.5769562e+00	 1.3424443e-01	  1.2106634e+00 	 2.1334291e-01
     125	 1.5237665e+00	 1.2958694e-01	 1.5781365e+00	 1.3421615e-01	  1.2091685e+00 	 1.7577434e-01


.. parsed-literal::

     126	 1.5261357e+00	 1.2941766e-01	 1.5805422e+00	 1.3413435e-01	  1.2000262e+00 	 2.0565557e-01
     127	 1.5271939e+00	 1.2939280e-01	 1.5816668e+00	 1.3415964e-01	  1.1927792e+00 	 1.7646313e-01


.. parsed-literal::

     128	 1.5286726e+00	 1.2933606e-01	 1.5831206e+00	 1.3411253e-01	  1.1911575e+00 	 2.0848274e-01
     129	 1.5301535e+00	 1.2929954e-01	 1.5846051e+00	 1.3409624e-01	  1.1875521e+00 	 1.7925286e-01


.. parsed-literal::

     130	 1.5312326e+00	 1.2929894e-01	 1.5856858e+00	 1.3408815e-01	  1.1867248e+00 	 1.8222189e-01


.. parsed-literal::

     131	 1.5336616e+00	 1.2926469e-01	 1.5881555e+00	 1.3406888e-01	  1.1815297e+00 	 2.0341492e-01


.. parsed-literal::

     132	 1.5350897e+00	 1.2920312e-01	 1.5896147e+00	 1.3406399e-01	  1.1803729e+00 	 3.1175470e-01


.. parsed-literal::

     133	 1.5367488e+00	 1.2917648e-01	 1.5912888e+00	 1.3405072e-01	  1.1790026e+00 	 2.0584702e-01
     134	 1.5382642e+00	 1.2909106e-01	 1.5928469e+00	 1.3392114e-01	  1.1749436e+00 	 1.8910408e-01


.. parsed-literal::

     135	 1.5394117e+00	 1.2903458e-01	 1.5940359e+00	 1.3378534e-01	  1.1759964e+00 	 1.9937134e-01


.. parsed-literal::

     136	 1.5406464e+00	 1.2900727e-01	 1.5952824e+00	 1.3371828e-01	  1.1727383e+00 	 2.2010517e-01
     137	 1.5422618e+00	 1.2901163e-01	 1.5969291e+00	 1.3365792e-01	  1.1640615e+00 	 1.7069173e-01


.. parsed-literal::

     138	 1.5435541e+00	 1.2907701e-01	 1.5982611e+00	 1.3369037e-01	  1.1596887e+00 	 2.1526647e-01


.. parsed-literal::

     139	 1.5451696e+00	 1.2912072e-01	 1.5999454e+00	 1.3376780e-01	  1.1487977e+00 	 2.0688605e-01


.. parsed-literal::

     140	 1.5461332e+00	 1.2920530e-01	 1.6009623e+00	 1.3392377e-01	  1.1394254e+00 	 2.0987678e-01


.. parsed-literal::

     141	 1.5472663e+00	 1.2913205e-01	 1.6020684e+00	 1.3388716e-01	  1.1425359e+00 	 2.0584726e-01
     142	 1.5479272e+00	 1.2905996e-01	 1.6027168e+00	 1.3386508e-01	  1.1442017e+00 	 1.9867849e-01


.. parsed-literal::

     143	 1.5489947e+00	 1.2899093e-01	 1.6037904e+00	 1.3387379e-01	  1.1433668e+00 	 2.0521998e-01
     144	 1.5505167e+00	 1.2888236e-01	 1.6053384e+00	 1.3392161e-01	  1.1338187e+00 	 1.8348145e-01


.. parsed-literal::

     145	 1.5517611e+00	 1.2887288e-01	 1.6066260e+00	 1.3396215e-01	  1.1264653e+00 	 2.0773816e-01


.. parsed-literal::

     146	 1.5528188e+00	 1.2885228e-01	 1.6076605e+00	 1.3395639e-01	  1.1238015e+00 	 2.0311451e-01
     147	 1.5540018e+00	 1.2883367e-01	 1.6088494e+00	 1.3394330e-01	  1.1180419e+00 	 1.9640732e-01


.. parsed-literal::

     148	 1.5548956e+00	 1.2876936e-01	 1.6097586e+00	 1.3390806e-01	  1.1158163e+00 	 2.1118236e-01


.. parsed-literal::

     149	 1.5560364e+00	 1.2863416e-01	 1.6110654e+00	 1.3377361e-01	  1.1056470e+00 	 2.1320271e-01


.. parsed-literal::

     150	 1.5581077e+00	 1.2845465e-01	 1.6130974e+00	 1.3369600e-01	  1.1030728e+00 	 2.1703577e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 1s, sys: 971 ms, total: 2min 2s
    Wall time: 30.6 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc2484b6410>



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
    CPU times: user 1.66 s, sys: 56 ms, total: 1.72 s
    Wall time: 524 ms


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

