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
       1	-3.3739427e-01	 3.1901776e-01	-3.2766264e-01	 3.2639274e-01	[-3.4234060e-01]	 4.5669699e-01


.. parsed-literal::

       2	-2.6766026e-01	 3.0838425e-01	-2.4382794e-01	 3.1633096e-01	[-2.6860847e-01]	 2.2640085e-01


.. parsed-literal::

       3	-2.2542447e-01	 2.8905135e-01	-1.8474723e-01	 2.9424584e-01	[-2.0947006e-01]	 2.7874875e-01
       4	-1.8938265e-01	 2.6544077e-01	-1.4869110e-01	 2.6902663e-01	[-1.7251538e-01]	 1.8843770e-01


.. parsed-literal::

       5	-1.0371481e-01	 2.5657504e-01	-6.9392811e-02	 2.6040499e-01	[-8.9681909e-02]	 2.0263362e-01


.. parsed-literal::

       6	-6.8978153e-02	 2.5126358e-01	-3.8498619e-02	 2.5429279e-01	[-5.0251047e-02]	 2.0395279e-01


.. parsed-literal::

       7	-5.1928093e-02	 2.4864770e-01	-2.7592894e-02	 2.5124109e-01	[-3.7834777e-02]	 2.0470810e-01
       8	-3.9006282e-02	 2.4651143e-01	-1.8689253e-02	 2.4845372e-01	[-2.6506189e-02]	 1.9042230e-01


.. parsed-literal::

       9	-2.5905306e-02	 2.4413476e-01	-8.5311850e-03	 2.4578223e-01	[-1.5403212e-02]	 2.1135521e-01


.. parsed-literal::

      10	-1.4364626e-02	 2.4196627e-01	 9.3177637e-04	 2.4413148e-01	[-7.9171635e-03]	 2.1022844e-01
      11	-1.1421586e-02	 2.4152591e-01	 2.4887475e-03	 2.4343248e-01	[-5.8671915e-03]	 1.7682505e-01


.. parsed-literal::

      12	-7.5179087e-03	 2.4089550e-01	 6.2356591e-03	 2.4256934e-01	[-4.4979481e-04]	 1.7996645e-01
      13	-5.1404626e-03	 2.4039100e-01	 8.5891356e-03	 2.4192519e-01	[ 2.3921537e-03]	 1.7037559e-01


.. parsed-literal::

      14	-1.5623920e-03	 2.3970422e-01	 1.2330024e-02	 2.4115666e-01	[ 6.4283661e-03]	 2.0688343e-01


.. parsed-literal::

      15	 1.0980111e-01	 2.2553148e-01	 1.2936662e-01	 2.2641608e-01	[ 1.2570398e-01]	 3.1364727e-01


.. parsed-literal::

      16	 1.4444995e-01	 2.2307539e-01	 1.6654663e-01	 2.2493146e-01	[ 1.6408864e-01]	 3.2065177e-01


.. parsed-literal::

      17	 2.0717304e-01	 2.1677179e-01	 2.3093444e-01	 2.1803275e-01	[ 2.2947845e-01]	 2.0347881e-01


.. parsed-literal::

      18	 3.0171862e-01	 2.1255404e-01	 3.3376784e-01	 2.1503048e-01	[ 3.2338823e-01]	 2.0923471e-01
      19	 3.4364575e-01	 2.0667332e-01	 3.7669242e-01	 2.1174162e-01	[ 3.5577586e-01]	 1.9595408e-01


.. parsed-literal::

      20	 3.8451188e-01	 2.0525085e-01	 4.1772326e-01	 2.1121760e-01	[ 3.9502708e-01]	 2.0589733e-01


.. parsed-literal::

      21	 4.2746509e-01	 2.0414286e-01	 4.6032299e-01	 2.0940643e-01	[ 4.4072487e-01]	 2.1647787e-01


.. parsed-literal::

      22	 4.7968819e-01	 2.0211800e-01	 5.1275362e-01	 2.0617900e-01	[ 4.9746478e-01]	 2.0634675e-01
      23	 6.0945110e-01	 2.0055118e-01	 6.4737627e-01	 2.0484990e-01	[ 6.2970130e-01]	 1.7600346e-01


.. parsed-literal::

      24	 6.2991326e-01	 2.0255316e-01	 6.7006855e-01	 2.0554621e-01	[ 6.5748202e-01]	 2.0380020e-01


.. parsed-literal::

      25	 6.8291772e-01	 1.9583853e-01	 7.2021714e-01	 1.9991546e-01	[ 7.0867452e-01]	 2.1502042e-01
      26	 7.1189321e-01	 1.9377496e-01	 7.4938079e-01	 1.9945331e-01	[ 7.3547046e-01]	 1.7944670e-01


.. parsed-literal::

      27	 7.3970513e-01	 1.9633511e-01	 7.7598546e-01	 2.0079358e-01	[ 7.6100993e-01]	 2.0096707e-01


.. parsed-literal::

      28	 7.8589225e-01	 1.9172064e-01	 8.2289637e-01	 1.9659268e-01	[ 7.9725144e-01]	 2.1403861e-01


.. parsed-literal::

      29	 8.0344507e-01	 1.8771543e-01	 8.4208135e-01	 1.9434824e-01	[ 8.1230904e-01]	 2.0897317e-01


.. parsed-literal::

      30	 8.2764046e-01	 1.8536755e-01	 8.6717297e-01	 1.9238299e-01	[ 8.3591574e-01]	 2.0984888e-01


.. parsed-literal::

      31	 8.5926010e-01	 1.7838902e-01	 8.9952906e-01	 1.8721623e-01	[ 8.6822042e-01]	 2.0894837e-01


.. parsed-literal::

      32	 8.9005334e-01	 1.7301557e-01	 9.3082887e-01	 1.8173066e-01	[ 8.9320882e-01]	 2.0889282e-01
      33	 9.1185878e-01	 1.6842985e-01	 9.5302516e-01	 1.7743958e-01	[ 9.2073956e-01]	 2.0005774e-01


.. parsed-literal::

      34	 9.3064895e-01	 1.6629891e-01	 9.7220867e-01	 1.7589065e-01	[ 9.3802751e-01]	 2.0841622e-01


.. parsed-literal::

      35	 9.5189006e-01	 1.6627906e-01	 9.9455574e-01	 1.7564293e-01	[ 9.4650834e-01]	 2.1357894e-01


.. parsed-literal::

      36	 9.6399832e-01	 1.6946153e-01	 1.0074999e+00	 1.7832456e-01	[ 9.5345716e-01]	 2.1841097e-01
      37	 9.7665070e-01	 1.6762527e-01	 1.0200689e+00	 1.7711878e-01	[ 9.6131951e-01]	 1.7194605e-01


.. parsed-literal::

      38	 9.8524998e-01	 1.6628384e-01	 1.0287000e+00	 1.7611288e-01	[ 9.6835332e-01]	 2.0817113e-01
      39	 9.9934557e-01	 1.6518560e-01	 1.0430076e+00	 1.7541087e-01	[ 9.8110639e-01]	 1.9879889e-01


.. parsed-literal::

      40	 1.0206687e+00	 1.6397380e-01	 1.0650174e+00	 1.7545691e-01	[ 1.0029954e+00]	 1.9602728e-01


.. parsed-literal::

      41	 1.0330929e+00	 1.6365303e-01	 1.0784270e+00	 1.7558616e-01	[ 1.0160112e+00]	 2.1621060e-01
      42	 1.0421795e+00	 1.6251115e-01	 1.0872057e+00	 1.7454451e-01	[ 1.0270345e+00]	 2.0851636e-01


.. parsed-literal::

      43	 1.0512454e+00	 1.6126116e-01	 1.0962525e+00	 1.7374081e-01	[ 1.0371199e+00]	 2.1258903e-01


.. parsed-literal::

      44	 1.0604011e+00	 1.6009121e-01	 1.1057418e+00	 1.7307308e-01	[ 1.0458399e+00]	 2.0619154e-01
      45	 1.0677126e+00	 1.5895518e-01	 1.1137643e+00	 1.7323077e-01	  1.0454890e+00 	 2.0008349e-01


.. parsed-literal::

      46	 1.0787815e+00	 1.5768549e-01	 1.1247080e+00	 1.7171521e-01	[ 1.0608304e+00]	 1.8962431e-01


.. parsed-literal::

      47	 1.0841669e+00	 1.5722710e-01	 1.1300775e+00	 1.7111707e-01	[ 1.0654528e+00]	 2.0808530e-01
      48	 1.0927618e+00	 1.5616351e-01	 1.1388774e+00	 1.6987376e-01	[ 1.0714331e+00]	 1.8373013e-01


.. parsed-literal::

      49	 1.1019852e+00	 1.5399909e-01	 1.1484174e+00	 1.6808768e-01	[ 1.0772552e+00]	 1.8589902e-01


.. parsed-literal::

      50	 1.1116412e+00	 1.5239401e-01	 1.1582389e+00	 1.6665832e-01	[ 1.0858858e+00]	 2.0193243e-01


.. parsed-literal::

      51	 1.1199076e+00	 1.5090970e-01	 1.1665432e+00	 1.6530678e-01	[ 1.0929461e+00]	 2.0283318e-01
      52	 1.1269706e+00	 1.4969054e-01	 1.1737103e+00	 1.6414665e-01	[ 1.0992626e+00]	 1.7669439e-01


.. parsed-literal::

      53	 1.1341860e+00	 1.4868870e-01	 1.1812956e+00	 1.6334401e-01	[ 1.0996007e+00]	 2.0392060e-01


.. parsed-literal::

      54	 1.1430595e+00	 1.4762186e-01	 1.1903007e+00	 1.6205028e-01	[ 1.1069078e+00]	 2.1581125e-01
      55	 1.1485659e+00	 1.4726478e-01	 1.1958195e+00	 1.6145233e-01	[ 1.1119736e+00]	 1.7209792e-01


.. parsed-literal::

      56	 1.1590956e+00	 1.4625542e-01	 1.2065899e+00	 1.6004885e-01	[ 1.1190472e+00]	 1.8668818e-01


.. parsed-literal::

      57	 1.1672749e+00	 1.4470730e-01	 1.2152908e+00	 1.5857603e-01	[ 1.1366280e+00]	 2.1375203e-01


.. parsed-literal::

      58	 1.1774792e+00	 1.4373948e-01	 1.2254476e+00	 1.5737342e-01	[ 1.1441537e+00]	 2.0705914e-01
      59	 1.1838287e+00	 1.4268459e-01	 1.2318860e+00	 1.5640046e-01	[ 1.1505812e+00]	 1.9705367e-01


.. parsed-literal::

      60	 1.1905329e+00	 1.4144353e-01	 1.2388390e+00	 1.5544501e-01	[ 1.1565069e+00]	 2.1608186e-01
      61	 1.1983085e+00	 1.4000078e-01	 1.2470346e+00	 1.5495089e-01	[ 1.1609894e+00]	 1.7968225e-01


.. parsed-literal::

      62	 1.2052091e+00	 1.3919063e-01	 1.2540235e+00	 1.5454378e-01	[ 1.1677905e+00]	 1.9571352e-01


.. parsed-literal::

      63	 1.2116374e+00	 1.3905335e-01	 1.2604033e+00	 1.5458905e-01	[ 1.1735021e+00]	 2.1172118e-01
      64	 1.2211366e+00	 1.3897880e-01	 1.2700101e+00	 1.5548972e-01	[ 1.1794942e+00]	 2.0525479e-01


.. parsed-literal::

      65	 1.2270090e+00	 1.3870046e-01	 1.2762623e+00	 1.5667969e-01	[ 1.1837671e+00]	 2.1077394e-01


.. parsed-literal::

      66	 1.2341216e+00	 1.3817362e-01	 1.2832615e+00	 1.5619544e-01	[ 1.1895446e+00]	 2.0950937e-01


.. parsed-literal::

      67	 1.2404147e+00	 1.3725704e-01	 1.2897669e+00	 1.5560534e-01	[ 1.1926800e+00]	 2.0739412e-01
      68	 1.2447813e+00	 1.3679794e-01	 1.2942613e+00	 1.5527819e-01	[ 1.1966052e+00]	 1.9085479e-01


.. parsed-literal::

      69	 1.2506828e+00	 1.3575278e-01	 1.3004947e+00	 1.5462380e-01	[ 1.1995954e+00]	 2.1875310e-01


.. parsed-literal::

      70	 1.2564711e+00	 1.3582616e-01	 1.3063462e+00	 1.5486191e-01	[ 1.2050087e+00]	 2.1998191e-01


.. parsed-literal::

      71	 1.2602627e+00	 1.3581944e-01	 1.3100063e+00	 1.5480317e-01	[ 1.2103400e+00]	 2.0785975e-01


.. parsed-literal::

      72	 1.2664647e+00	 1.3553232e-01	 1.3164157e+00	 1.5455919e-01	[ 1.2131171e+00]	 2.1258736e-01


.. parsed-literal::

      73	 1.2735742e+00	 1.3537252e-01	 1.3239083e+00	 1.5488272e-01	[ 1.2186492e+00]	 2.0930123e-01


.. parsed-literal::

      74	 1.2799142e+00	 1.3484094e-01	 1.3304114e+00	 1.5401381e-01	[ 1.2243570e+00]	 2.0798016e-01
      75	 1.2855643e+00	 1.3448222e-01	 1.3360754e+00	 1.5372115e-01	[ 1.2292305e+00]	 1.8452048e-01


.. parsed-literal::

      76	 1.2917364e+00	 1.3405385e-01	 1.3424663e+00	 1.5350741e-01	[ 1.2350175e+00]	 2.0745254e-01


.. parsed-literal::

      77	 1.2967169e+00	 1.3353784e-01	 1.3477045e+00	 1.5246986e-01	[ 1.2395312e+00]	 2.1013904e-01


.. parsed-literal::

      78	 1.3024681e+00	 1.3327493e-01	 1.3535281e+00	 1.5223186e-01	[ 1.2462742e+00]	 2.0974946e-01
      79	 1.3074788e+00	 1.3292978e-01	 1.3586878e+00	 1.5170544e-01	[ 1.2522746e+00]	 1.9853759e-01


.. parsed-literal::

      80	 1.3118728e+00	 1.3263609e-01	 1.3631778e+00	 1.5136613e-01	[ 1.2567340e+00]	 1.9963837e-01


.. parsed-literal::

      81	 1.3183944e+00	 1.3249030e-01	 1.3699362e+00	 1.5085818e-01	[ 1.2613220e+00]	 2.0413375e-01
      82	 1.3244454e+00	 1.3207479e-01	 1.3759911e+00	 1.5079889e-01	[ 1.2677662e+00]	 1.8470573e-01


.. parsed-literal::

      83	 1.3274724e+00	 1.3198016e-01	 1.3788873e+00	 1.5095856e-01	[ 1.2700985e+00]	 2.0458794e-01


.. parsed-literal::

      84	 1.3321841e+00	 1.3161832e-01	 1.3836734e+00	 1.5092694e-01	[ 1.2712535e+00]	 2.0219827e-01
      85	 1.3380779e+00	 1.3128974e-01	 1.3898267e+00	 1.5100586e-01	[ 1.2737443e+00]	 1.8341899e-01


.. parsed-literal::

      86	 1.3435485e+00	 1.3059712e-01	 1.3954379e+00	 1.5071780e-01	[ 1.2782593e+00]	 1.7701077e-01


.. parsed-literal::

      87	 1.3479726e+00	 1.3048133e-01	 1.3998902e+00	 1.5057027e-01	[ 1.2821598e+00]	 2.1021175e-01


.. parsed-literal::

      88	 1.3531882e+00	 1.3010303e-01	 1.4051211e+00	 1.5025325e-01	[ 1.2870749e+00]	 2.0388722e-01
      89	 1.3572261e+00	 1.3002698e-01	 1.4092572e+00	 1.4995074e-01	[ 1.2894712e+00]	 1.9709992e-01


.. parsed-literal::

      90	 1.3615365e+00	 1.2971869e-01	 1.4136506e+00	 1.4981184e-01	[ 1.2917923e+00]	 2.0769477e-01


.. parsed-literal::

      91	 1.3651295e+00	 1.2930206e-01	 1.4172679e+00	 1.4955952e-01	[ 1.2955210e+00]	 2.9524899e-01


.. parsed-literal::

      92	 1.3691340e+00	 1.2921119e-01	 1.4213597e+00	 1.4977218e-01	[ 1.2964699e+00]	 2.1170235e-01
      93	 1.3734351e+00	 1.2922635e-01	 1.4257048e+00	 1.5006652e-01	[ 1.2989984e+00]	 2.0292592e-01


.. parsed-literal::

      94	 1.3787258e+00	 1.2932582e-01	 1.4311720e+00	 1.5056474e-01	[ 1.3011293e+00]	 2.0686817e-01


.. parsed-literal::

      95	 1.3801694e+00	 1.2962482e-01	 1.4329626e+00	 1.5116774e-01	  1.3007295e+00 	 2.0418167e-01


.. parsed-literal::

      96	 1.3847212e+00	 1.2939932e-01	 1.4373478e+00	 1.5095929e-01	[ 1.3056311e+00]	 2.1347094e-01


.. parsed-literal::

      97	 1.3866752e+00	 1.2930715e-01	 1.4393190e+00	 1.5084914e-01	[ 1.3079475e+00]	 2.1703720e-01
      98	 1.3897052e+00	 1.2919593e-01	 1.4424302e+00	 1.5068427e-01	[ 1.3122818e+00]	 2.0322037e-01


.. parsed-literal::

      99	 1.3940933e+00	 1.2899891e-01	 1.4469324e+00	 1.5036169e-01	[ 1.3177576e+00]	 2.0990634e-01


.. parsed-literal::

     100	 1.3970865e+00	 1.2909199e-01	 1.4502430e+00	 1.5036352e-01	[ 1.3245474e+00]	 2.1374893e-01


.. parsed-literal::

     101	 1.4011722e+00	 1.2887524e-01	 1.4541680e+00	 1.5018382e-01	[ 1.3254030e+00]	 2.0931292e-01
     102	 1.4032985e+00	 1.2867252e-01	 1.4562622e+00	 1.5018850e-01	  1.3243898e+00 	 1.8575764e-01


.. parsed-literal::

     103	 1.4063902e+00	 1.2836230e-01	 1.4594215e+00	 1.5022631e-01	  1.3235644e+00 	 2.0783472e-01


.. parsed-literal::

     104	 1.4097039e+00	 1.2771023e-01	 1.4629820e+00	 1.5058201e-01	  1.3219499e+00 	 2.0197368e-01
     105	 1.4139153e+00	 1.2755366e-01	 1.4672170e+00	 1.5051185e-01	[ 1.3260798e+00]	 1.9978428e-01


.. parsed-literal::

     106	 1.4162030e+00	 1.2752688e-01	 1.4694842e+00	 1.5040940e-01	[ 1.3299335e+00]	 2.1818471e-01


.. parsed-literal::

     107	 1.4200189e+00	 1.2740195e-01	 1.4733616e+00	 1.5026204e-01	[ 1.3336872e+00]	 2.1105647e-01


.. parsed-literal::

     108	 1.4245888e+00	 1.2721763e-01	 1.4780036e+00	 1.4992439e-01	[ 1.3410696e+00]	 2.0627475e-01
     109	 1.4278882e+00	 1.2681086e-01	 1.4815168e+00	 1.4984557e-01	  1.3345341e+00 	 1.7902565e-01


.. parsed-literal::

     110	 1.4311308e+00	 1.2663607e-01	 1.4845971e+00	 1.4961246e-01	  1.3397236e+00 	 2.1539164e-01
     111	 1.4330918e+00	 1.2636306e-01	 1.4864936e+00	 1.4944290e-01	[ 1.3414013e+00]	 2.0440888e-01


.. parsed-literal::

     112	 1.4361222e+00	 1.2589526e-01	 1.4896265e+00	 1.4918795e-01	  1.3404762e+00 	 2.1282125e-01


.. parsed-literal::

     113	 1.4400296e+00	 1.2543866e-01	 1.4936863e+00	 1.4901144e-01	  1.3408836e+00 	 2.1923399e-01
     114	 1.4433405e+00	 1.2510548e-01	 1.4971232e+00	 1.4867330e-01	  1.3398668e+00 	 1.7781520e-01


.. parsed-literal::

     115	 1.4464735e+00	 1.2493327e-01	 1.5003563e+00	 1.4879377e-01	  1.3403401e+00 	 2.1010160e-01
     116	 1.4491488e+00	 1.2471308e-01	 1.5030469e+00	 1.4862922e-01	[ 1.3419759e+00]	 1.9174981e-01


.. parsed-literal::

     117	 1.4518691e+00	 1.2456043e-01	 1.5057874e+00	 1.4886576e-01	[ 1.3422289e+00]	 2.1492457e-01


.. parsed-literal::

     118	 1.4543995e+00	 1.2438196e-01	 1.5083504e+00	 1.4900703e-01	[ 1.3425229e+00]	 2.1207595e-01
     119	 1.4571700e+00	 1.2402190e-01	 1.5111862e+00	 1.4896637e-01	  1.3419735e+00 	 1.8020153e-01


.. parsed-literal::

     120	 1.4598695e+00	 1.2388163e-01	 1.5139233e+00	 1.4893485e-01	  1.3419912e+00 	 2.1256089e-01
     121	 1.4627423e+00	 1.2365597e-01	 1.5168940e+00	 1.4872393e-01	  1.3424752e+00 	 1.9560480e-01


.. parsed-literal::

     122	 1.4657159e+00	 1.2355392e-01	 1.5199652e+00	 1.4855853e-01	  1.3402807e+00 	 1.8653679e-01


.. parsed-literal::

     123	 1.4684643e+00	 1.2349427e-01	 1.5227155e+00	 1.4857162e-01	[ 1.3437209e+00]	 2.0310688e-01
     124	 1.4705213e+00	 1.2353726e-01	 1.5246772e+00	 1.4866561e-01	[ 1.3467268e+00]	 2.0704770e-01


.. parsed-literal::

     125	 1.4731042e+00	 1.2356818e-01	 1.5272101e+00	 1.4903647e-01	[ 1.3508665e+00]	 2.1900272e-01


.. parsed-literal::

     126	 1.4738731e+00	 1.2370883e-01	 1.5280810e+00	 1.4917612e-01	[ 1.3521119e+00]	 2.0736051e-01
     127	 1.4760262e+00	 1.2360178e-01	 1.5301782e+00	 1.4919109e-01	[ 1.3527890e+00]	 1.9956017e-01


.. parsed-literal::

     128	 1.4773063e+00	 1.2350712e-01	 1.5315067e+00	 1.4919417e-01	  1.3520218e+00 	 2.0408845e-01
     129	 1.4788022e+00	 1.2340259e-01	 1.5330942e+00	 1.4918940e-01	  1.3506920e+00 	 1.9898319e-01


.. parsed-literal::

     130	 1.4814231e+00	 1.2323393e-01	 1.5358169e+00	 1.4911053e-01	  1.3489947e+00 	 2.1731663e-01


.. parsed-literal::

     131	 1.4828768e+00	 1.2316250e-01	 1.5373476e+00	 1.4921739e-01	  1.3484437e+00 	 2.8878474e-01


.. parsed-literal::

     132	 1.4851506e+00	 1.2308698e-01	 1.5396425e+00	 1.4911510e-01	  1.3479733e+00 	 2.0375133e-01


.. parsed-literal::

     133	 1.4869760e+00	 1.2302666e-01	 1.5414293e+00	 1.4907365e-01	  1.3499130e+00 	 2.1310496e-01
     134	 1.4895697e+00	 1.2300535e-01	 1.5439956e+00	 1.4928694e-01	  1.3505262e+00 	 1.6868639e-01


.. parsed-literal::

     135	 1.4908257e+00	 1.2282226e-01	 1.5453699e+00	 1.4945041e-01	  1.3508755e+00 	 2.2832227e-01


.. parsed-literal::

     136	 1.4929798e+00	 1.2283077e-01	 1.5474514e+00	 1.4960609e-01	  1.3505522e+00 	 2.0867586e-01
     137	 1.4946589e+00	 1.2278510e-01	 1.5491672e+00	 1.4979597e-01	  1.3475730e+00 	 1.7958665e-01


.. parsed-literal::

     138	 1.4962406e+00	 1.2269897e-01	 1.5508206e+00	 1.4992720e-01	  1.3441218e+00 	 2.0928001e-01


.. parsed-literal::

     139	 1.4982868e+00	 1.2253771e-01	 1.5529911e+00	 1.4984186e-01	  1.3385083e+00 	 2.0834851e-01


.. parsed-literal::

     140	 1.5002163e+00	 1.2235166e-01	 1.5550294e+00	 1.4992139e-01	  1.3340049e+00 	 2.0227504e-01
     141	 1.5014536e+00	 1.2230862e-01	 1.5561998e+00	 1.4979451e-01	  1.3379222e+00 	 1.8163347e-01


.. parsed-literal::

     142	 1.5025247e+00	 1.2222913e-01	 1.5572162e+00	 1.4963573e-01	  1.3406643e+00 	 2.0279598e-01


.. parsed-literal::

     143	 1.5040814e+00	 1.2202636e-01	 1.5587995e+00	 1.4963353e-01	  1.3405311e+00 	 2.0280266e-01
     144	 1.5059772e+00	 1.2176189e-01	 1.5607525e+00	 1.4952000e-01	  1.3394205e+00 	 1.9003987e-01


.. parsed-literal::

     145	 1.5074800e+00	 1.2160772e-01	 1.5623367e+00	 1.4958196e-01	  1.3376261e+00 	 1.9881916e-01


.. parsed-literal::

     146	 1.5094553e+00	 1.2143894e-01	 1.5644834e+00	 1.4961683e-01	  1.3328318e+00 	 2.0868945e-01


.. parsed-literal::

     147	 1.5113052e+00	 1.2131882e-01	 1.5664679e+00	 1.4955158e-01	  1.3319541e+00 	 2.0739913e-01
     148	 1.5132274e+00	 1.2123840e-01	 1.5684007e+00	 1.4932918e-01	  1.3324743e+00 	 2.0126700e-01


.. parsed-literal::

     149	 1.5148677e+00	 1.2111710e-01	 1.5700319e+00	 1.4900839e-01	  1.3334696e+00 	 2.1495438e-01
     150	 1.5160369e+00	 1.2103715e-01	 1.5711365e+00	 1.4882665e-01	  1.3376026e+00 	 1.8701220e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.04 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f89d0488d30>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.04 s, sys: 34 ms, total: 2.07 s
    Wall time: 617 ms


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

