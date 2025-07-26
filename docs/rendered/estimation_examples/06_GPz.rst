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
       1	-3.5054966e-01	 3.2298175e-01	-3.4073771e-01	 3.1209154e-01	[-3.1839920e-01]	 4.5732188e-01


.. parsed-literal::

       2	-2.7787892e-01	 3.1117498e-01	-2.5342309e-01	 3.0559029e-01	[-2.2762059e-01]	 2.2881937e-01


.. parsed-literal::

       3	-2.3622027e-01	 2.9254061e-01	-1.9652291e-01	 2.8617476e-01	[-1.6406195e-01]	 3.0109787e-01
       4	-1.9693930e-01	 2.6738067e-01	-1.5630539e-01	 2.6098521e-01	[-1.1729693e-01]	 1.9239569e-01


.. parsed-literal::

       5	-1.1348401e-01	 2.5882155e-01	-7.8900905e-02	 2.5722574e-01	[-5.3038476e-02]	 2.1644998e-01
       6	-7.6771534e-02	 2.5285904e-01	-4.5115088e-02	 2.4648976e-01	[-2.0342270e-02]	 1.9807291e-01


.. parsed-literal::

       7	-5.9538315e-02	 2.5042522e-01	-3.4775713e-02	 2.4414064e-01	[-8.2364171e-03]	 2.0481610e-01


.. parsed-literal::

       8	-4.4754666e-02	 2.4790560e-01	-2.4313197e-02	 2.4168516e-01	[ 2.4205422e-03]	 2.1250820e-01


.. parsed-literal::

       9	-3.1061170e-02	 2.4537985e-01	-1.3861501e-02	 2.3852185e-01	[ 1.5582223e-02]	 2.0542598e-01


.. parsed-literal::

      10	-2.6582395e-02	 2.4452609e-01	-1.1326753e-02	 2.3751116e-01	  1.4097710e-02 	 2.1469140e-01
      11	-1.7886081e-02	 2.4308341e-01	-3.1711379e-03	 2.3684208e-01	[ 2.2688601e-02]	 1.8475103e-01


.. parsed-literal::

      12	-1.4866736e-02	 2.4244187e-01	-3.4138135e-04	 2.3637822e-01	[ 2.5385302e-02]	 1.7648411e-01


.. parsed-literal::

      13	-1.1565782e-02	 2.4177317e-01	 2.8516758e-03	 2.3567272e-01	[ 2.9330885e-02]	 2.0337677e-01


.. parsed-literal::

      14	-2.2902644e-03	 2.3989710e-01	 1.2822248e-02	 2.3459054e-01	[ 3.6121260e-02]	 2.0782733e-01


.. parsed-literal::

      15	 5.0693870e-02	 2.3176916e-01	 6.8742910e-02	 2.3062100e-01	[ 8.7843899e-02]	 2.0384288e-01


.. parsed-literal::

      16	 9.5998134e-02	 2.2306524e-01	 1.1607169e-01	 2.2501356e-01	[ 1.2311439e-01]	 3.2647395e-01
      17	 1.5715481e-01	 2.1918887e-01	 1.7958178e-01	 2.1953215e-01	[ 1.8214871e-01]	 2.0015240e-01


.. parsed-literal::

      18	 2.6026026e-01	 2.1438761e-01	 2.9218995e-01	 2.1686195e-01	[ 2.8292261e-01]	 1.7060709e-01
      19	 3.0323584e-01	 2.1337853e-01	 3.3559648e-01	 2.1337691e-01	[ 3.3539301e-01]	 1.8909144e-01


.. parsed-literal::

      20	 3.4962659e-01	 2.1035317e-01	 3.8149016e-01	 2.0794045e-01	[ 3.8240701e-01]	 2.0926547e-01


.. parsed-literal::

      21	 3.8609831e-01	 2.0703551e-01	 4.1848959e-01	 2.0565492e-01	[ 4.1561682e-01]	 2.0125461e-01
      22	 4.3088977e-01	 2.0426892e-01	 4.6483916e-01	 2.0559335e-01	[ 4.5761328e-01]	 1.7697763e-01


.. parsed-literal::

      23	 5.1438354e-01	 2.0407418e-01	 5.5540610e-01	 2.0562580e-01	[ 5.4767725e-01]	 1.8900037e-01


.. parsed-literal::

      24	 5.6085072e-01	 2.0221662e-01	 6.0252279e-01	 2.0483855e-01	[ 6.0086568e-01]	 2.1133375e-01


.. parsed-literal::

      25	 5.9336250e-01	 2.0019779e-01	 6.3331826e-01	 2.0255578e-01	[ 6.3058581e-01]	 2.0512319e-01


.. parsed-literal::

      26	 6.3983926e-01	 1.9702625e-01	 6.8000269e-01	 1.9740262e-01	[ 6.7971804e-01]	 2.0849419e-01
      27	 6.7632837e-01	 1.9701261e-01	 7.1413715e-01	 1.9630389e-01	[ 7.1304941e-01]	 1.7817259e-01


.. parsed-literal::

      28	 7.0747514e-01	 1.9954556e-01	 7.4548500e-01	 1.9652887e-01	[ 7.4459243e-01]	 2.0997667e-01


.. parsed-literal::

      29	 7.4379775e-01	 1.9916970e-01	 7.8246887e-01	 1.9553231e-01	[ 7.7728286e-01]	 2.0510864e-01


.. parsed-literal::

      30	 7.6634209e-01	 2.0532137e-01	 8.0494345e-01	 2.0921397e-01	[ 7.9431294e-01]	 2.0588565e-01
      31	 7.8881210e-01	 2.0861353e-01	 8.3142237e-01	 1.9738421e-01	[ 8.2429713e-01]	 1.7862868e-01


.. parsed-literal::

      32	 8.2951303e-01	 2.0207401e-01	 8.7003745e-01	 1.9546405e-01	[ 8.6514714e-01]	 2.0434380e-01


.. parsed-literal::

      33	 8.4815128e-01	 1.9914392e-01	 8.8855526e-01	 1.9749267e-01	[ 8.7678898e-01]	 2.0799899e-01


.. parsed-literal::

      34	 8.6652571e-01	 1.9760775e-01	 9.0775492e-01	 2.0745634e-01	  8.6819539e-01 	 2.0340204e-01


.. parsed-literal::

      35	 8.9002953e-01	 1.9753824e-01	 9.3102577e-01	 1.9970389e-01	[ 9.0199521e-01]	 2.0997882e-01
      36	 9.0221105e-01	 1.9648395e-01	 9.4340120e-01	 1.9547066e-01	[ 9.1593484e-01]	 1.7972422e-01


.. parsed-literal::

      37	 9.3558863e-01	 1.9353656e-01	 9.7812491e-01	 1.8584597e-01	[ 9.4790064e-01]	 1.8648767e-01


.. parsed-literal::

      38	 9.4819980e-01	 1.9087233e-01	 9.9247259e-01	 1.8813412e-01	[ 9.6492070e-01]	 2.0776224e-01
      39	 9.7403786e-01	 1.8901715e-01	 1.0182091e+00	 1.9029071e-01	[ 9.8644887e-01]	 1.9518614e-01


.. parsed-literal::

      40	 9.8499429e-01	 1.8778446e-01	 1.0295716e+00	 1.9123745e-01	[ 9.9372023e-01]	 2.1451616e-01


.. parsed-literal::

      41	 1.0010571e+00	 1.8498717e-01	 1.0463369e+00	 1.9182256e-01	[ 1.0058687e+00]	 2.0154667e-01


.. parsed-literal::

      42	 1.0211019e+00	 1.8119571e-01	 1.0670896e+00	 1.8477348e-01	[ 1.0257562e+00]	 2.0716929e-01
      43	 1.0333652e+00	 1.7940602e-01	 1.0802256e+00	 1.7860716e-01	[ 1.0445435e+00]	 2.0104480e-01


.. parsed-literal::

      44	 1.0492282e+00	 1.7852163e-01	 1.0957072e+00	 1.7669842e-01	[ 1.0547152e+00]	 1.9659925e-01


.. parsed-literal::

      45	 1.0583470e+00	 1.7814447e-01	 1.1047630e+00	 1.7493973e-01	[ 1.0580839e+00]	 2.0643306e-01


.. parsed-literal::

      46	 1.0685136e+00	 1.7618540e-01	 1.1149700e+00	 1.7226889e-01	[ 1.0639084e+00]	 2.0426488e-01
      47	 1.0861832e+00	 1.7138083e-01	 1.1329582e+00	 1.6748065e-01	[ 1.0717016e+00]	 1.9902277e-01


.. parsed-literal::

      48	 1.1068670e+00	 1.6274023e-01	 1.1536691e+00	 1.5959922e-01	[ 1.0872672e+00]	 2.0692205e-01


.. parsed-literal::

      49	 1.1224847e+00	 1.5583588e-01	 1.1699049e+00	 1.5511566e-01	[ 1.0948605e+00]	 2.1665287e-01


.. parsed-literal::

      50	 1.1348969e+00	 1.5376905e-01	 1.1828430e+00	 1.5306531e-01	[ 1.1079093e+00]	 2.1628928e-01


.. parsed-literal::

      51	 1.1461954e+00	 1.5214016e-01	 1.1940559e+00	 1.5186042e-01	[ 1.1197190e+00]	 2.0584416e-01


.. parsed-literal::

      52	 1.1569517e+00	 1.5081040e-01	 1.2050782e+00	 1.5048274e-01	[ 1.1308498e+00]	 2.1547127e-01
      53	 1.1684619e+00	 1.4883580e-01	 1.2168454e+00	 1.4837423e-01	[ 1.1423984e+00]	 1.9535708e-01


.. parsed-literal::

      54	 1.1836170e+00	 1.4663048e-01	 1.2327925e+00	 1.4407553e-01	[ 1.1510042e+00]	 2.1027184e-01


.. parsed-literal::

      55	 1.1964396e+00	 1.4659916e-01	 1.2456480e+00	 1.4411284e-01	[ 1.1690852e+00]	 2.1832085e-01


.. parsed-literal::

      56	 1.2037497e+00	 1.4491412e-01	 1.2527457e+00	 1.4357851e-01	[ 1.1759376e+00]	 2.1082973e-01


.. parsed-literal::

      57	 1.2117950e+00	 1.4483999e-01	 1.2608437e+00	 1.4373188e-01	[ 1.1799834e+00]	 2.0839763e-01
      58	 1.2209621e+00	 1.4369096e-01	 1.2706329e+00	 1.4369470e-01	[ 1.1820149e+00]	 1.9719744e-01


.. parsed-literal::

      59	 1.2320514e+00	 1.4306191e-01	 1.2816002e+00	 1.4397370e-01	[ 1.1976819e+00]	 2.0072770e-01


.. parsed-literal::

      60	 1.2434920e+00	 1.4170626e-01	 1.2936943e+00	 1.4341658e-01	[ 1.2022635e+00]	 2.0978498e-01
      61	 1.2512802e+00	 1.4027228e-01	 1.3017801e+00	 1.4287680e-01	[ 1.2091794e+00]	 1.8641281e-01


.. parsed-literal::

      62	 1.2579004e+00	 1.3962862e-01	 1.3081547e+00	 1.4233758e-01	[ 1.2125367e+00]	 2.1407938e-01


.. parsed-literal::

      63	 1.2647480e+00	 1.3907422e-01	 1.3148922e+00	 1.4196236e-01	[ 1.2166107e+00]	 2.1568894e-01
      64	 1.2722315e+00	 1.3835906e-01	 1.3226444e+00	 1.4217251e-01	[ 1.2173743e+00]	 1.7702317e-01


.. parsed-literal::

      65	 1.2801622e+00	 1.3767802e-01	 1.3304263e+00	 1.4167162e-01	[ 1.2254357e+00]	 1.7606592e-01


.. parsed-literal::

      66	 1.2890785e+00	 1.3682763e-01	 1.3395430e+00	 1.4144479e-01	[ 1.2304394e+00]	 2.0716834e-01
      67	 1.2997332e+00	 1.3516686e-01	 1.3506289e+00	 1.4001296e-01	[ 1.2361664e+00]	 1.9451523e-01


.. parsed-literal::

      68	 1.3084287e+00	 1.3353662e-01	 1.3595455e+00	 1.3916887e-01	[ 1.2417446e+00]	 2.0992279e-01
      69	 1.3153895e+00	 1.3247544e-01	 1.3666601e+00	 1.3794505e-01	[ 1.2432188e+00]	 1.7858982e-01


.. parsed-literal::

      70	 1.3240814e+00	 1.3141372e-01	 1.3754934e+00	 1.3636486e-01	[ 1.2507555e+00]	 2.1301126e-01
      71	 1.3323641e+00	 1.3025495e-01	 1.3837955e+00	 1.3497144e-01	[ 1.2541536e+00]	 1.9529748e-01


.. parsed-literal::

      72	 1.3411443e+00	 1.2930830e-01	 1.3927437e+00	 1.3390738e-01	[ 1.2641870e+00]	 1.9158745e-01
      73	 1.3467812e+00	 1.2882119e-01	 1.3982397e+00	 1.3254982e-01	[ 1.2715837e+00]	 1.7273879e-01


.. parsed-literal::

      74	 1.3522657e+00	 1.2863731e-01	 1.4037488e+00	 1.3266767e-01	[ 1.2745426e+00]	 1.9445348e-01


.. parsed-literal::

      75	 1.3607712e+00	 1.2804571e-01	 1.4127691e+00	 1.3242828e-01	[ 1.2787612e+00]	 2.0807004e-01
      76	 1.3665868e+00	 1.2783703e-01	 1.4186076e+00	 1.3179142e-01	[ 1.2805727e+00]	 1.9958138e-01


.. parsed-literal::

      77	 1.3720108e+00	 1.2761481e-01	 1.4238769e+00	 1.3116800e-01	[ 1.2901361e+00]	 2.0288634e-01
      78	 1.3790707e+00	 1.2754783e-01	 1.4310940e+00	 1.2963098e-01	[ 1.2996689e+00]	 1.9940233e-01


.. parsed-literal::

      79	 1.3836511e+00	 1.2728706e-01	 1.4359355e+00	 1.2904631e-01	[ 1.3039731e+00]	 2.0735908e-01


.. parsed-literal::

      80	 1.3880251e+00	 1.2717706e-01	 1.4402823e+00	 1.2871098e-01	[ 1.3053917e+00]	 2.0344543e-01


.. parsed-literal::

      81	 1.3963285e+00	 1.2692357e-01	 1.4487261e+00	 1.2761935e-01	[ 1.3059580e+00]	 2.1066904e-01
      82	 1.4011303e+00	 1.2645804e-01	 1.4535065e+00	 1.2729383e-01	[ 1.3064930e+00]	 1.6725326e-01


.. parsed-literal::

      83	 1.4065815e+00	 1.2611749e-01	 1.4588769e+00	 1.2681264e-01	[ 1.3104678e+00]	 2.0272183e-01


.. parsed-literal::

      84	 1.4127525e+00	 1.2581504e-01	 1.4650298e+00	 1.2618440e-01	[ 1.3227097e+00]	 2.0996761e-01
      85	 1.4164800e+00	 1.2560062e-01	 1.4689413e+00	 1.2607201e-01	  1.3193951e+00 	 1.8876171e-01


.. parsed-literal::

      86	 1.4202820e+00	 1.2550063e-01	 1.4727257e+00	 1.2612456e-01	[ 1.3243895e+00]	 2.1444511e-01
      87	 1.4295159e+00	 1.2543102e-01	 1.4822029e+00	 1.2596885e-01	[ 1.3344036e+00]	 1.8105578e-01


.. parsed-literal::

      88	 1.4310481e+00	 1.2510776e-01	 1.4838703e+00	 1.2565921e-01	[ 1.3393273e+00]	 1.7697954e-01


.. parsed-literal::

      89	 1.4349774e+00	 1.2497179e-01	 1.4876370e+00	 1.2549302e-01	[ 1.3413807e+00]	 2.1051598e-01
      90	 1.4375344e+00	 1.2488035e-01	 1.4902162e+00	 1.2509851e-01	[ 1.3418668e+00]	 1.9936132e-01


.. parsed-literal::

      91	 1.4406423e+00	 1.2470660e-01	 1.4934433e+00	 1.2455788e-01	[ 1.3419914e+00]	 1.7225790e-01
      92	 1.4452462e+00	 1.2463374e-01	 1.4982482e+00	 1.2418325e-01	  1.3407234e+00 	 1.7400527e-01


.. parsed-literal::

      93	 1.4476543e+00	 1.2467453e-01	 1.5010543e+00	 1.2417105e-01	  1.3407077e+00 	 2.0671630e-01


.. parsed-literal::

      94	 1.4537216e+00	 1.2468420e-01	 1.5068986e+00	 1.2414611e-01	[ 1.3428035e+00]	 2.0706010e-01
      95	 1.4558477e+00	 1.2456516e-01	 1.5088963e+00	 1.2432076e-01	[ 1.3454230e+00]	 1.9896889e-01


.. parsed-literal::

      96	 1.4599297e+00	 1.2452706e-01	 1.5129765e+00	 1.2467659e-01	[ 1.3488078e+00]	 1.9504428e-01
      97	 1.4629722e+00	 1.2446141e-01	 1.5161991e+00	 1.2475356e-01	  1.3414929e+00 	 1.9620347e-01


.. parsed-literal::

      98	 1.4661859e+00	 1.2438652e-01	 1.5193677e+00	 1.2455621e-01	  1.3454666e+00 	 2.0038891e-01


.. parsed-literal::

      99	 1.4691299e+00	 1.2441938e-01	 1.5223783e+00	 1.2426457e-01	  1.3466662e+00 	 2.2775173e-01


.. parsed-literal::

     100	 1.4720004e+00	 1.2447537e-01	 1.5252422e+00	 1.2416455e-01	  1.3478620e+00 	 2.1689034e-01
     101	 1.4756845e+00	 1.2469677e-01	 1.5290714e+00	 1.2380623e-01	[ 1.3562448e+00]	 2.0255446e-01


.. parsed-literal::

     102	 1.4801076e+00	 1.2476735e-01	 1.5334325e+00	 1.2422691e-01	  1.3510969e+00 	 2.0921969e-01
     103	 1.4818161e+00	 1.2462233e-01	 1.5350933e+00	 1.2429828e-01	  1.3544157e+00 	 1.8892026e-01


.. parsed-literal::

     104	 1.4847008e+00	 1.2447610e-01	 1.5381498e+00	 1.2462479e-01	  1.3540579e+00 	 1.9696760e-01
     105	 1.4869825e+00	 1.2449889e-01	 1.5405939e+00	 1.2500007e-01	[ 1.3566935e+00]	 2.0602822e-01


.. parsed-literal::

     106	 1.4890061e+00	 1.2454670e-01	 1.5426354e+00	 1.2506899e-01	[ 1.3568803e+00]	 2.0759320e-01
     107	 1.4921520e+00	 1.2460334e-01	 1.5458362e+00	 1.2544133e-01	[ 1.3569009e+00]	 1.9494915e-01


.. parsed-literal::

     108	 1.4939487e+00	 1.2462156e-01	 1.5475767e+00	 1.2567425e-01	[ 1.3596823e+00]	 2.1025467e-01


.. parsed-literal::

     109	 1.4971868e+00	 1.2442040e-01	 1.5507710e+00	 1.2636708e-01	[ 1.3638249e+00]	 2.1055388e-01


.. parsed-literal::

     110	 1.4991645e+00	 1.2425542e-01	 1.5528290e+00	 1.2719660e-01	  1.3605650e+00 	 2.1759701e-01
     111	 1.5016122e+00	 1.2411751e-01	 1.5552804e+00	 1.2703702e-01	  1.3623914e+00 	 1.7441249e-01


.. parsed-literal::

     112	 1.5042447e+00	 1.2393393e-01	 1.5580572e+00	 1.2680819e-01	  1.3594314e+00 	 2.0804882e-01


.. parsed-literal::

     113	 1.5063110e+00	 1.2384301e-01	 1.5601839e+00	 1.2652278e-01	  1.3591279e+00 	 2.1080542e-01


.. parsed-literal::

     114	 1.5078383e+00	 1.2359339e-01	 1.5619984e+00	 1.2572917e-01	  1.3519753e+00 	 2.0743680e-01


.. parsed-literal::

     115	 1.5112865e+00	 1.2379013e-01	 1.5652212e+00	 1.2565354e-01	  1.3628130e+00 	 2.1818161e-01


.. parsed-literal::

     116	 1.5123840e+00	 1.2382297e-01	 1.5662440e+00	 1.2578119e-01	[ 1.3667035e+00]	 2.0674443e-01


.. parsed-literal::

     117	 1.5149923e+00	 1.2380072e-01	 1.5688712e+00	 1.2562009e-01	[ 1.3703673e+00]	 2.1641564e-01


.. parsed-literal::

     118	 1.5164365e+00	 1.2384234e-01	 1.5703557e+00	 1.2540938e-01	[ 1.3717710e+00]	 3.1537914e-01


.. parsed-literal::

     119	 1.5180161e+00	 1.2375465e-01	 1.5720393e+00	 1.2511211e-01	  1.3684201e+00 	 2.1766615e-01
     120	 1.5200631e+00	 1.2363887e-01	 1.5742567e+00	 1.2474654e-01	  1.3609863e+00 	 1.9372058e-01


.. parsed-literal::

     121	 1.5224271e+00	 1.2356498e-01	 1.5767689e+00	 1.2444500e-01	  1.3540203e+00 	 2.0415211e-01


.. parsed-literal::

     122	 1.5237201e+00	 1.2353860e-01	 1.5781855e+00	 1.2425597e-01	  1.3490352e+00 	 3.1675339e-01


.. parsed-literal::

     123	 1.5253895e+00	 1.2350037e-01	 1.5798521e+00	 1.2423684e-01	  1.3502171e+00 	 2.1440077e-01
     124	 1.5266376e+00	 1.2348932e-01	 1.5810373e+00	 1.2433246e-01	  1.3530669e+00 	 2.0094013e-01


.. parsed-literal::

     125	 1.5286051e+00	 1.2339397e-01	 1.5829650e+00	 1.2436170e-01	  1.3569643e+00 	 2.1628070e-01
     126	 1.5302525e+00	 1.2332051e-01	 1.5846495e+00	 1.2471316e-01	  1.3535731e+00 	 1.9790530e-01


.. parsed-literal::

     127	 1.5318345e+00	 1.2330635e-01	 1.5862524e+00	 1.2456111e-01	  1.3561107e+00 	 2.0072627e-01
     128	 1.5336917e+00	 1.2317069e-01	 1.5882249e+00	 1.2424231e-01	  1.3517165e+00 	 1.9915700e-01


.. parsed-literal::

     129	 1.5350153e+00	 1.2311583e-01	 1.5895961e+00	 1.2418170e-01	  1.3524987e+00 	 2.0388508e-01


.. parsed-literal::

     130	 1.5366591e+00	 1.2299980e-01	 1.5912532e+00	 1.2400377e-01	  1.3535471e+00 	 2.0366144e-01
     131	 1.5380003e+00	 1.2287745e-01	 1.5925758e+00	 1.2361727e-01	  1.3543655e+00 	 2.0123219e-01


.. parsed-literal::

     132	 1.5391623e+00	 1.2284336e-01	 1.5936258e+00	 1.2355001e-01	  1.3611068e+00 	 1.9587231e-01


.. parsed-literal::

     133	 1.5403452e+00	 1.2279336e-01	 1.5946970e+00	 1.2331867e-01	  1.3675796e+00 	 2.0360684e-01


.. parsed-literal::

     134	 1.5424244e+00	 1.2269012e-01	 1.5966986e+00	 1.2294747e-01	[ 1.3739371e+00]	 2.0700836e-01


.. parsed-literal::

     135	 1.5432868e+00	 1.2258129e-01	 1.5975560e+00	 1.2259973e-01	[ 1.3810772e+00]	 2.1224427e-01
     136	 1.5445538e+00	 1.2258903e-01	 1.5988406e+00	 1.2258391e-01	  1.3779828e+00 	 1.8948603e-01


.. parsed-literal::

     137	 1.5453632e+00	 1.2258942e-01	 1.5997233e+00	 1.2258063e-01	  1.3744547e+00 	 2.0366621e-01
     138	 1.5466002e+00	 1.2258076e-01	 1.6010585e+00	 1.2251504e-01	  1.3707950e+00 	 1.7828441e-01


.. parsed-literal::

     139	 1.5478892e+00	 1.2264668e-01	 1.6024837e+00	 1.2251271e-01	  1.3657332e+00 	 1.9812036e-01


.. parsed-literal::

     140	 1.5494270e+00	 1.2261000e-01	 1.6040108e+00	 1.2249165e-01	  1.3656653e+00 	 2.1533847e-01


.. parsed-literal::

     141	 1.5504444e+00	 1.2257823e-01	 1.6049723e+00	 1.2245132e-01	  1.3686243e+00 	 2.1376204e-01
     142	 1.5517346e+00	 1.2254499e-01	 1.6062438e+00	 1.2250029e-01	  1.3707904e+00 	 1.9692278e-01


.. parsed-literal::

     143	 1.5525477e+00	 1.2254838e-01	 1.6070878e+00	 1.2242777e-01	  1.3707599e+00 	 3.1538868e-01


.. parsed-literal::

     144	 1.5537924e+00	 1.2253192e-01	 1.6083718e+00	 1.2252668e-01	  1.3713077e+00 	 2.0965290e-01


.. parsed-literal::

     145	 1.5547498e+00	 1.2250398e-01	 1.6094216e+00	 1.2251231e-01	  1.3685398e+00 	 2.0925283e-01


.. parsed-literal::

     146	 1.5558608e+00	 1.2250145e-01	 1.6107115e+00	 1.2245351e-01	  1.3672079e+00 	 2.0247531e-01


.. parsed-literal::

     147	 1.5569572e+00	 1.2242245e-01	 1.6118948e+00	 1.2219301e-01	  1.3636234e+00 	 2.0815372e-01


.. parsed-literal::

     148	 1.5579201e+00	 1.2235722e-01	 1.6128936e+00	 1.2192543e-01	  1.3631918e+00 	 2.1002126e-01


.. parsed-literal::

     149	 1.5591463e+00	 1.2227566e-01	 1.6141829e+00	 1.2155889e-01	  1.3629944e+00 	 2.0472479e-01
     150	 1.5607793e+00	 1.2221668e-01	 1.6159158e+00	 1.2111898e-01	  1.3643113e+00 	 1.8463111e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.09 s, total: 2min 4s
    Wall time: 31.3 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f4c209a8ac0>



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
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 1.67 s, sys: 47 ms, total: 1.72 s
    Wall time: 523 ms


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

