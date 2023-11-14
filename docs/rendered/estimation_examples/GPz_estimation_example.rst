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
    from rail.core.utils import find_rail_file
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
    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.0524725e-01	 3.2336423e-01	-2.9133691e-01	 2.9721600e-01	[-2.2070101e-01]	 6.7829657e-01
       2	-2.4010339e-01	 2.9500899e-01	-2.0739165e-01	 2.7843489e-01	[-1.4165536e-01]	 1.4613247e-01
       3	-1.5323885e-01	 2.6883054e-01	-1.1938429e-01	 2.6117549e-01	[-8.2972544e-02]	 1.5332222e-01
       4	-9.3152239e-02	 2.5951483e-01	-7.0005523e-02	 2.4903957e-01	[-3.0588142e-02]	 1.4814067e-01
       5	-6.7083706e-02	 2.5302719e-01	-4.5199831e-02	 2.4130661e-01	[-4.7498175e-04]	 1.4607668e-01
       6	-4.5264956e-02	 2.4839507e-01	-2.7117616e-02	 2.3613870e-01	[ 2.2555210e-02]	 1.4888859e-01
       7	-3.7375808e-02	 2.4575698e-01	-2.0500529e-02	 2.3486541e-01	  1.6640796e-02 	 1.5064549e-01
       8	-2.6224671e-02	 2.4442743e-01	-1.0942357e-02	 2.3451695e-01	[ 3.0313593e-02]	 1.5210700e-01
       9	-1.8558864e-02	 2.4274284e-01	-3.1320426e-03	 2.3256649e-01	[ 3.7439297e-02]	 1.4743304e-01
      10	-1.4999562e-02	 2.4187975e-01	 5.9621023e-04	 2.3238508e-01	[ 3.8135728e-02]	 1.5120339e-01
      11	-8.2393991e-03	 2.4039685e-01	 7.0774954e-03	 2.3250154e-01	  3.8056195e-02 	 1.5318656e-01
      12	 1.2115000e-03	 2.3793914e-01	 1.7664882e-02	 2.3114612e-01	[ 4.4381842e-02]	 1.4571881e-01
      13	 2.6519266e-02	 2.3277901e-01	 4.4980194e-02	 2.2667339e-01	[ 6.8886091e-02]	 1.4645648e-01
      14	 1.1279985e-01	 2.1819752e-01	 1.3918756e-01	 2.1663370e-01	[ 1.4510420e-01]	 1.5053821e-01
      15	 1.4749027e-01	 2.1757630e-01	 1.7683575e-01	 2.1643324e-01	[ 1.8190193e-01]	 3.0322957e-01
      16	 2.1006090e-01	 2.0708055e-01	 2.4125481e-01	 2.0392129e-01	[ 2.4719970e-01]	 1.4820480e-01
      17	 2.9111770e-01	 2.0120657e-01	 3.2495132e-01	 1.9943181e-01	[ 3.3288045e-01]	 1.4842796e-01
      18	 3.2333675e-01	 1.9736984e-01	 3.5992526e-01	 1.9396215e-01	[ 3.7511953e-01]	 1.4720011e-01
      19	 3.6775038e-01	 1.9287256e-01	 4.0450037e-01	 1.9175779e-01	[ 4.1000512e-01]	 1.4851141e-01
      20	 4.0660627e-01	 1.8866398e-01	 4.4391315e-01	 1.8686225e-01	[ 4.4986354e-01]	 1.5093946e-01
      21	 4.7889397e-01	 1.8273681e-01	 5.1730284e-01	 1.8244777e-01	[ 5.1984172e-01]	 1.5014672e-01
      22	 6.0047642e-01	 1.7943104e-01	 6.4102517e-01	 1.7884232e-01	[ 6.4003929e-01]	 1.5070772e-01
      23	 6.3592612e-01	 1.8245331e-01	 6.7888380e-01	 1.8184969e-01	[ 6.7631646e-01]	 1.5081501e-01
      24	 6.5645858e-01	 1.7634858e-01	 6.9546041e-01	 1.7835433e-01	[ 6.8812483e-01]	 1.4786410e-01
      25	 7.0583194e-01	 1.7173605e-01	 7.4670851e-01	 1.7514640e-01	[ 7.3786938e-01]	 1.4936709e-01
      26	 7.2216072e-01	 1.6964285e-01	 7.6428746e-01	 1.7136570e-01	[ 7.6278513e-01]	 1.5151787e-01
      27	 7.4909880e-01	 1.6645990e-01	 7.9058195e-01	 1.6909674e-01	[ 7.7945378e-01]	 1.4395022e-01
      28	 7.6990375e-01	 1.6484591e-01	 8.1083665e-01	 1.6786913e-01	[ 8.0207874e-01]	 1.4915299e-01
      29	 7.9503368e-01	 1.6495146e-01	 8.3645273e-01	 1.6480604e-01	[ 8.3366122e-01]	 1.5020895e-01
      30	 8.1578524e-01	 1.6195340e-01	 8.5761660e-01	 1.6400935e-01	[ 8.4813425e-01]	 1.4768815e-01
      31	 8.3855112e-01	 1.6026074e-01	 8.8084152e-01	 1.6200890e-01	[ 8.6500414e-01]	 1.4632392e-01
      32	 8.7013325e-01	 1.5669120e-01	 9.1310383e-01	 1.5827664e-01	[ 8.9196967e-01]	 1.4556623e-01
      33	 9.0211328e-01	 1.5472794e-01	 9.4617315e-01	 1.5312684e-01	[ 9.3447127e-01]	 1.5233994e-01
      34	 9.2645221e-01	 1.5305534e-01	 9.7030340e-01	 1.5118547e-01	[ 9.6221305e-01]	 1.4617586e-01
      35	 9.4306491e-01	 1.5140109e-01	 9.8655959e-01	 1.5046090e-01	[ 9.7608537e-01]	 1.4673018e-01
      36	 9.6229121e-01	 1.4935115e-01	 1.0059293e+00	 1.4985989e-01	[ 9.9266943e-01]	 1.4609742e-01
      37	 9.7441093e-01	 1.4961999e-01	 1.0184971e+00	 1.5032469e-01	[ 1.0038606e+00]	 1.4922667e-01
      38	 9.8775945e-01	 1.4832611e-01	 1.0322468e+00	 1.5052122e-01	[ 1.0170730e+00]	 1.4729834e-01
      39	 1.0047435e+00	 1.4706722e-01	 1.0501308e+00	 1.5039864e-01	[ 1.0339725e+00]	 1.4857960e-01
      40	 1.0191228e+00	 1.4621040e-01	 1.0652460e+00	 1.4893492e-01	[ 1.0492355e+00]	 1.5248156e-01
      41	 1.0338909e+00	 1.4667528e-01	 1.0805988e+00	 1.4953319e-01	[ 1.0637502e+00]	 1.4990449e-01
      42	 1.0473242e+00	 1.4605771e-01	 1.0940000e+00	 1.4768632e-01	[ 1.0769379e+00]	 1.4781642e-01
      43	 1.0667451e+00	 1.4573145e-01	 1.1136911e+00	 1.4493037e-01	[ 1.0968644e+00]	 1.5151048e-01
      44	 1.0850405e+00	 1.4536854e-01	 1.1334622e+00	 1.4104335e-01	[ 1.1122468e+00]	 1.4578867e-01
      45	 1.0967800e+00	 1.4425572e-01	 1.1458258e+00	 1.3913359e-01	[ 1.1234120e+00]	 1.4627719e-01
      46	 1.1060093e+00	 1.4310122e-01	 1.1549064e+00	 1.3871593e-01	[ 1.1312643e+00]	 1.4377904e-01
      47	 1.1119526e+00	 1.4248246e-01	 1.1610198e+00	 1.3845117e-01	[ 1.1368221e+00]	 1.5505385e-01
      48	 1.1210698e+00	 1.4176619e-01	 1.1703718e+00	 1.3833747e-01	[ 1.1460115e+00]	 1.4488316e-01
      49	 1.1347893e+00	 1.4096271e-01	 1.1844270e+00	 1.3757364e-01	[ 1.1576491e+00]	 1.5192032e-01
      50	 1.1371801e+00	 1.4064408e-01	 1.1877043e+00	 1.4095972e-01	  1.1439995e+00 	 1.4959407e-01
      51	 1.1551487e+00	 1.3921142e-01	 1.2050260e+00	 1.3839581e-01	[ 1.1678140e+00]	 1.5388012e-01
      52	 1.1605796e+00	 1.3880241e-01	 1.2104655e+00	 1.3764021e-01	[ 1.1727035e+00]	 1.4599252e-01
      53	 1.1696136e+00	 1.3811526e-01	 1.2197270e+00	 1.3718895e-01	[ 1.1777065e+00]	 1.5393043e-01
      54	 1.1815727e+00	 1.3706522e-01	 1.2319825e+00	 1.3699140e-01	[ 1.1852903e+00]	 1.4785814e-01
      55	 1.1897044e+00	 1.3668323e-01	 1.2407262e+00	 1.3796679e-01	[ 1.1920116e+00]	 1.5017033e-01
      56	 1.2006554e+00	 1.3586295e-01	 1.2514403e+00	 1.3713538e-01	[ 1.2035973e+00]	 1.5043068e-01
      57	 1.2093856e+00	 1.3509694e-01	 1.2601792e+00	 1.3614449e-01	[ 1.2131317e+00]	 1.4976144e-01
      58	 1.2195791e+00	 1.3432767e-01	 1.2705180e+00	 1.3482110e-01	[ 1.2220915e+00]	 1.4580369e-01
      59	 1.2257585e+00	 1.3516641e-01	 1.2770473e+00	 1.3356785e-01	[ 1.2351434e+00]	 1.5042901e-01
      60	 1.2345056e+00	 1.3428539e-01	 1.2856996e+00	 1.3326422e-01	[ 1.2402415e+00]	 1.5760040e-01
      61	 1.2401999e+00	 1.3401032e-01	 1.2915078e+00	 1.3319396e-01	[ 1.2419072e+00]	 1.5176034e-01
      62	 1.2480791e+00	 1.3402366e-01	 1.2995748e+00	 1.3299463e-01	[ 1.2460391e+00]	 1.4806485e-01
      63	 1.2603250e+00	 1.3381602e-01	 1.3120749e+00	 1.3318573e-01	[ 1.2562591e+00]	 1.4909005e-01
      64	 1.2670011e+00	 1.3436046e-01	 1.3192544e+00	 1.3266709e-01	[ 1.2607635e+00]	 1.4710808e-01
      65	 1.2751141e+00	 1.3323846e-01	 1.3272067e+00	 1.3216984e-01	[ 1.2709211e+00]	 1.5077567e-01
      66	 1.2806317e+00	 1.3230368e-01	 1.3327112e+00	 1.3180195e-01	[ 1.2781884e+00]	 1.4610577e-01
      67	 1.2878296e+00	 1.3141160e-01	 1.3401482e+00	 1.3159939e-01	[ 1.2867245e+00]	 2.6987028e-01
      68	 1.2980189e+00	 1.3103514e-01	 1.3506184e+00	 1.3120143e-01	[ 1.2979332e+00]	 1.4969254e-01
      69	 1.3037368e+00	 1.3055078e-01	 1.3570993e+00	 1.3113699e-01	  1.2938501e+00 	 1.4806390e-01
      70	 1.3127097e+00	 1.3035648e-01	 1.3657779e+00	 1.3082764e-01	[ 1.3028452e+00]	 1.4921665e-01
      71	 1.3190222e+00	 1.3024680e-01	 1.3721747e+00	 1.3016916e-01	[ 1.3057264e+00]	 1.5112376e-01
      72	 1.3265806e+00	 1.2996609e-01	 1.3800853e+00	 1.2961498e-01	[ 1.3069327e+00]	 1.4906001e-01
      73	 1.3328277e+00	 1.2913651e-01	 1.3867156e+00	 1.2848736e-01	[ 1.3085491e+00]	 1.5889287e-01
      74	 1.3415588e+00	 1.2860841e-01	 1.3955207e+00	 1.2887457e-01	[ 1.3131256e+00]	 1.4713240e-01
      75	 1.3476392e+00	 1.2837758e-01	 1.4016582e+00	 1.2951042e-01	[ 1.3173549e+00]	 1.5218854e-01
      76	 1.3546922e+00	 1.2825298e-01	 1.4088668e+00	 1.3013691e-01	[ 1.3207313e+00]	 1.4816380e-01
      77	 1.3627073e+00	 1.2821132e-01	 1.4172067e+00	 1.3004250e-01	[ 1.3214590e+00]	 1.4976740e-01
      78	 1.3694786e+00	 1.2827570e-01	 1.4240891e+00	 1.2931335e-01	[ 1.3223084e+00]	 1.5012193e-01
      79	 1.3743698e+00	 1.2805688e-01	 1.4287763e+00	 1.2834042e-01	[ 1.3276853e+00]	 1.8301272e-01
      80	 1.3823834e+00	 1.2822379e-01	 1.4369392e+00	 1.2674573e-01	  1.3269418e+00 	 1.4756250e-01
      81	 1.3880539e+00	 1.2871504e-01	 1.4426586e+00	 1.2569888e-01	[ 1.3331560e+00]	 1.5271926e-01
      82	 1.3942803e+00	 1.2873224e-01	 1.4488739e+00	 1.2537876e-01	[ 1.3352160e+00]	 1.4989877e-01
      83	 1.4017707e+00	 1.2867519e-01	 1.4565053e+00	 1.2563380e-01	[ 1.3362100e+00]	 1.4893627e-01
      84	 1.4072622e+00	 1.2808828e-01	 1.4622370e+00	 1.2554612e-01	[ 1.3375980e+00]	 1.4922690e-01
      85	 1.4131740e+00	 1.2753799e-01	 1.4683722e+00	 1.2534065e-01	[ 1.3424501e+00]	 1.4868116e-01
      86	 1.4184964e+00	 1.2720461e-01	 1.4739640e+00	 1.2495933e-01	[ 1.3465672e+00]	 1.5123606e-01
      87	 1.4226416e+00	 1.2693489e-01	 1.4780772e+00	 1.2449390e-01	[ 1.3514817e+00]	 1.4647913e-01
      88	 1.4281239e+00	 1.2699616e-01	 1.4836215e+00	 1.2373725e-01	[ 1.3534141e+00]	 1.5055346e-01
      89	 1.4328694e+00	 1.2717187e-01	 1.4884066e+00	 1.2337191e-01	[ 1.3556636e+00]	 1.4714670e-01
      90	 1.4372225e+00	 1.2703481e-01	 1.4927649e+00	 1.2288070e-01	[ 1.3567256e+00]	 1.5212822e-01
      91	 1.4408003e+00	 1.2664929e-01	 1.4964036e+00	 1.2258846e-01	[ 1.3578738e+00]	 1.4619923e-01
      92	 1.4442111e+00	 1.2628836e-01	 1.5000129e+00	 1.2239951e-01	  1.3561568e+00 	 1.4620113e-01
      93	 1.4484170e+00	 1.2594327e-01	 1.5042652e+00	 1.2214442e-01	[ 1.3599642e+00]	 1.5218663e-01
      94	 1.4525039e+00	 1.2548807e-01	 1.5083257e+00	 1.2203140e-01	[ 1.3661938e+00]	 1.5045238e-01
      95	 1.4557464e+00	 1.2497112e-01	 1.5115231e+00	 1.2169234e-01	[ 1.3712880e+00]	 1.4796591e-01
      96	 1.4594545e+00	 1.2478340e-01	 1.5151498e+00	 1.2165899e-01	[ 1.3763453e+00]	 1.4637923e-01
      97	 1.4629737e+00	 1.2468828e-01	 1.5186896e+00	 1.2167049e-01	[ 1.3786478e+00]	 1.5318894e-01
      98	 1.4672973e+00	 1.2458194e-01	 1.5231590e+00	 1.2172728e-01	[ 1.3791048e+00]	 1.5025949e-01
      99	 1.4703750e+00	 1.2451517e-01	 1.5263723e+00	 1.2184895e-01	[ 1.3802805e+00]	 3.0188918e-01
     100	 1.4743670e+00	 1.2440878e-01	 1.5304774e+00	 1.2180780e-01	  1.3800271e+00 	 1.5349579e-01
     101	 1.4770427e+00	 1.2419042e-01	 1.5331859e+00	 1.2169799e-01	[ 1.3815862e+00]	 1.5650725e-01
     102	 1.4794491e+00	 1.2404959e-01	 1.5356309e+00	 1.2158190e-01	[ 1.3862017e+00]	 1.5203118e-01
     103	 1.4819271e+00	 1.2385366e-01	 1.5380435e+00	 1.2141082e-01	[ 1.3882506e+00]	 1.4935493e-01
     104	 1.4836849e+00	 1.2371041e-01	 1.5398014e+00	 1.2136809e-01	[ 1.3893669e+00]	 1.4620233e-01
     105	 1.4863598e+00	 1.2341278e-01	 1.5425649e+00	 1.2134584e-01	  1.3890835e+00 	 1.5182543e-01
     106	 1.4884227e+00	 1.2295977e-01	 1.5448538e+00	 1.2168592e-01	  1.3848033e+00 	 1.6503310e-01
     107	 1.4915326e+00	 1.2274533e-01	 1.5479735e+00	 1.2157175e-01	  1.3839341e+00 	 1.6003561e-01
     108	 1.4931394e+00	 1.2261415e-01	 1.5496012e+00	 1.2158735e-01	  1.3834421e+00 	 1.4970326e-01
     109	 1.4953326e+00	 1.2239744e-01	 1.5518370e+00	 1.2161384e-01	  1.3826612e+00 	 1.5181184e-01
     110	 1.4992841e+00	 1.2199598e-01	 1.5558500e+00	 1.2141275e-01	  1.3848528e+00 	 1.4718962e-01
     111	 1.5015013e+00	 1.2174401e-01	 1.5581946e+00	 1.2166219e-01	  1.3798189e+00 	 3.0320597e-01
     112	 1.5046607e+00	 1.2141116e-01	 1.5613616e+00	 1.2137955e-01	  1.3849758e+00 	 1.5042138e-01
     113	 1.5071678e+00	 1.2113479e-01	 1.5638955e+00	 1.2119411e-01	  1.3879365e+00 	 1.5458369e-01
     114	 1.5096299e+00	 1.2056900e-01	 1.5665126e+00	 1.2119800e-01	[ 1.3908364e+00]	 1.4726114e-01
     115	 1.5121058e+00	 1.2032713e-01	 1.5690201e+00	 1.2128696e-01	  1.3905575e+00 	 1.4850879e-01
     116	 1.5138609e+00	 1.2014548e-01	 1.5707990e+00	 1.2144348e-01	  1.3889210e+00 	 1.4960217e-01
     117	 1.5160706e+00	 1.1987843e-01	 1.5730373e+00	 1.2184889e-01	  1.3883162e+00 	 1.5190840e-01
     118	 1.5182445e+00	 1.1968341e-01	 1.5752380e+00	 1.2196077e-01	  1.3885787e+00 	 1.4588213e-01
     119	 1.5207024e+00	 1.1956866e-01	 1.5776815e+00	 1.2223358e-01	[ 1.3927178e+00]	 1.5254045e-01
     120	 1.5229006e+00	 1.1944051e-01	 1.5799493e+00	 1.2238495e-01	[ 1.3968006e+00]	 1.4973187e-01
     121	 1.5245672e+00	 1.1932225e-01	 1.5816206e+00	 1.2258353e-01	[ 1.3980288e+00]	 1.4947629e-01
     122	 1.5258137e+00	 1.1923374e-01	 1.5828930e+00	 1.2255989e-01	  1.3974565e+00 	 1.5016270e-01
     123	 1.5278397e+00	 1.1898086e-01	 1.5850370e+00	 1.2269232e-01	  1.3933025e+00 	 1.4935350e-01
     124	 1.5295125e+00	 1.1882094e-01	 1.5867740e+00	 1.2289485e-01	  1.3914296e+00 	 1.5409589e-01
     125	 1.5322250e+00	 1.1836479e-01	 1.5896295e+00	 1.2333778e-01	  1.3864266e+00 	 1.5028596e-01
     126	 1.5339341e+00	 1.1816237e-01	 1.5913505e+00	 1.2330687e-01	  1.3887272e+00 	 1.4992929e-01
     127	 1.5355172e+00	 1.1811266e-01	 1.5928755e+00	 1.2319953e-01	  1.3903718e+00 	 1.5462089e-01
     128	 1.5374757e+00	 1.1793279e-01	 1.5948521e+00	 1.2308999e-01	  1.3899312e+00 	 1.4756823e-01
     129	 1.5387742e+00	 1.1777690e-01	 1.5962456e+00	 1.2313251e-01	  1.3846357e+00 	 1.4857674e-01
     130	 1.5404294e+00	 1.1758782e-01	 1.5980141e+00	 1.2333405e-01	  1.3801140e+00 	 1.5141153e-01
     131	 1.5422518e+00	 1.1736606e-01	 1.5999538e+00	 1.2364641e-01	  1.3750695e+00 	 1.4671564e-01
     132	 1.5438826e+00	 1.1721481e-01	 1.6016324e+00	 1.2402159e-01	  1.3724838e+00 	 1.4919329e-01
     133	 1.5454223e+00	 1.1710173e-01	 1.6031375e+00	 1.2426318e-01	  1.3758644e+00 	 1.5205431e-01
     134	 1.5471692e+00	 1.1696677e-01	 1.6048536e+00	 1.2445459e-01	  1.3763647e+00 	 1.4579463e-01
     135	 1.5485637e+00	 1.1680547e-01	 1.6062724e+00	 1.2463706e-01	  1.3765393e+00 	 1.4754081e-01
     136	 1.5497403e+00	 1.1680547e-01	 1.6074291e+00	 1.2444639e-01	  1.3760473e+00 	 1.4591265e-01
     137	 1.5508568e+00	 1.1677729e-01	 1.6085858e+00	 1.2428405e-01	  1.3701400e+00 	 1.5591979e-01
     138	 1.5518309e+00	 1.1674844e-01	 1.6095965e+00	 1.2422433e-01	  1.3675132e+00 	 1.5016508e-01
     139	 1.5536224e+00	 1.1670293e-01	 1.6114630e+00	 1.2425473e-01	  1.3575464e+00 	 1.5261722e-01
     140	 1.5548874e+00	 1.1667987e-01	 1.6127725e+00	 1.2435061e-01	  1.3538557e+00 	 1.4982557e-01
     141	 1.5559629e+00	 1.1665783e-01	 1.6138708e+00	 1.2456436e-01	  1.3507802e+00 	 1.5017033e-01
     142	 1.5570215e+00	 1.1663571e-01	 1.6149493e+00	 1.2474838e-01	  1.3456902e+00 	 1.4699459e-01
     143	 1.5580659e+00	 1.1660162e-01	 1.6159911e+00	 1.2489499e-01	  1.3432147e+00 	 1.5016341e-01
     144	 1.5595025e+00	 1.1657221e-01	 1.6174552e+00	 1.2521249e-01	  1.3366867e+00 	 1.5126824e-01
     145	 1.5603766e+00	 1.1650669e-01	 1.6183781e+00	 1.2561507e-01	  1.3260264e+00 	 1.5052819e-01
     146	 1.5617065e+00	 1.1646223e-01	 1.6196636e+00	 1.2562430e-01	  1.3286409e+00 	 1.4994192e-01
     147	 1.5625965e+00	 1.1643736e-01	 1.6205575e+00	 1.2576458e-01	  1.3263967e+00 	 1.4979649e-01
     148	 1.5633789e+00	 1.1640321e-01	 1.6213564e+00	 1.2592354e-01	  1.3222262e+00 	 1.4764762e-01
     149	 1.5646319e+00	 1.1625043e-01	 1.6226938e+00	 1.2626011e-01	  1.3109261e+00 	 1.4825726e-01
     150	 1.5656975e+00	 1.1615714e-01	 1.6238590e+00	 1.2640394e-01	  1.2994214e+00 	 1.4675593e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 36.3 s, sys: 11.1 s, total: 47.4 s
    Wall time: 24 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f8b627f94b0>



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
    Process 0 running estimator on chunk 0 - 10000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000
    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run
    Process 0 running estimator on chunk 10000 - 20000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000
    Process 0 running estimator on chunk 20000 - 20449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 707 ms, sys: 304 ms, total: 1.01 s
    Wall time: 602 ms


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_16_1.png


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




.. image:: ../../../docs/rendered/estimation_examples/GPz_estimation_example_files/../../../docs/rendered/estimation_examples/GPz_estimation_example_19_1.png

