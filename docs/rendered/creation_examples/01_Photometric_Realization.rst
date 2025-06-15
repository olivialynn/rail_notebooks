Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fe4ae8bb0a0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.008200  0.006069  
    1      25.391064  0.056676  0.049429  
    2      24.304707  0.014065  0.010573  
    3      25.291103  0.114245  0.094595  
    4      25.096743  0.011587  0.010297  
    ...          ...       ...       ...  
    99995  24.737946  0.030005  0.016119  
    99996  24.224169  0.183101  0.100577  
    99997  25.613836  0.064761  0.040176  
    99998  25.274899  0.124022  0.104577  
    99999  25.699642  0.124695  0.099984  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.743132</td>
          <td>0.454044</td>
          <td>27.099180</td>
          <td>0.229984</td>
          <td>26.097366</td>
          <td>0.085915</td>
          <td>25.171131</td>
          <td>0.061794</td>
          <td>24.649748</td>
          <td>0.074533</td>
          <td>24.007660</td>
          <td>0.095150</td>
          <td>0.008200</td>
          <td>0.006069</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.303826</td>
          <td>1.257328</td>
          <td>27.362287</td>
          <td>0.285291</td>
          <td>26.747924</td>
          <td>0.151370</td>
          <td>26.013888</td>
          <td>0.129566</td>
          <td>25.716039</td>
          <td>0.188007</td>
          <td>26.018852</td>
          <td>0.498085</td>
          <td>0.056676</td>
          <td>0.049429</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.483846</td>
          <td>0.600783</td>
          <td>25.878225</td>
          <td>0.115166</td>
          <td>24.925991</td>
          <td>0.095072</td>
          <td>24.190171</td>
          <td>0.111625</td>
          <td>0.014065</td>
          <td>0.010573</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.273275</td>
          <td>0.665361</td>
          <td>30.118861</td>
          <td>1.703612</td>
          <td>27.491647</td>
          <td>0.281921</td>
          <td>26.339893</td>
          <td>0.171410</td>
          <td>25.545866</td>
          <td>0.162714</td>
          <td>25.011552</td>
          <td>0.225066</td>
          <td>0.114245</td>
          <td>0.094595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.005290</td>
          <td>0.253937</td>
          <td>26.164160</td>
          <td>0.103464</td>
          <td>25.884749</td>
          <td>0.071210</td>
          <td>25.550649</td>
          <td>0.086435</td>
          <td>25.312645</td>
          <td>0.133168</td>
          <td>24.993897</td>
          <td>0.221787</td>
          <td>0.011587</td>
          <td>0.010297</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.953155</td>
          <td>0.530382</td>
          <td>26.321168</td>
          <td>0.118635</td>
          <td>25.415757</td>
          <td>0.046974</td>
          <td>25.068020</td>
          <td>0.056391</td>
          <td>24.911807</td>
          <td>0.093895</td>
          <td>24.505472</td>
          <td>0.146676</td>
          <td>0.030005</td>
          <td>0.016119</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.861223</td>
          <td>0.495809</td>
          <td>26.786096</td>
          <td>0.176876</td>
          <td>26.263729</td>
          <td>0.099438</td>
          <td>25.159114</td>
          <td>0.061139</td>
          <td>24.813542</td>
          <td>0.086122</td>
          <td>24.065794</td>
          <td>0.100126</td>
          <td>0.183101</td>
          <td>0.100577</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.175770</td>
          <td>1.922359</td>
          <td>27.033445</td>
          <td>0.217760</td>
          <td>26.270336</td>
          <td>0.100015</td>
          <td>26.733486</td>
          <td>0.238461</td>
          <td>25.984885</td>
          <td>0.235386</td>
          <td>25.760179</td>
          <td>0.409952</td>
          <td>0.064761</td>
          <td>0.040176</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.301631</td>
          <td>0.322585</td>
          <td>26.275818</td>
          <td>0.114047</td>
          <td>26.231407</td>
          <td>0.096659</td>
          <td>25.872529</td>
          <td>0.114596</td>
          <td>25.491050</td>
          <td>0.155264</td>
          <td>25.222736</td>
          <td>0.267805</td>
          <td>0.124022</td>
          <td>0.104577</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.452967</td>
          <td>0.751331</td>
          <td>26.777940</td>
          <td>0.175657</td>
          <td>26.513043</td>
          <td>0.123596</td>
          <td>26.360296</td>
          <td>0.174409</td>
          <td>26.017028</td>
          <td>0.241719</td>
          <td>25.601531</td>
          <td>0.362539</td>
          <td>0.124695</td>
          <td>0.099984</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>26.912453</td>
          <td>0.567197</td>
          <td>26.793551</td>
          <td>0.204185</td>
          <td>25.883540</td>
          <td>0.083708</td>
          <td>25.091704</td>
          <td>0.068291</td>
          <td>24.767706</td>
          <td>0.097245</td>
          <td>23.997732</td>
          <td>0.111420</td>
          <td>0.008200</td>
          <td>0.006069</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.639522</td>
          <td>0.927493</td>
          <td>27.511864</td>
          <td>0.368503</td>
          <td>26.634065</td>
          <td>0.162215</td>
          <td>26.062785</td>
          <td>0.160903</td>
          <td>25.885574</td>
          <td>0.254568</td>
          <td>25.623678</td>
          <td>0.430635</td>
          <td>0.056676</td>
          <td>0.049429</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.380543</td>
          <td>0.691918</td>
          <td>29.626872</td>
          <td>1.362941</td>
          <td>25.990231</td>
          <td>0.149843</td>
          <td>25.084015</td>
          <td>0.128161</td>
          <td>24.417325</td>
          <td>0.160140</td>
          <td>0.014065</td>
          <td>0.010573</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.961371</td>
          <td>0.528280</td>
          <td>27.034960</td>
          <td>0.233134</td>
          <td>26.153019</td>
          <td>0.178419</td>
          <td>25.275640</td>
          <td>0.156504</td>
          <td>25.270540</td>
          <td>0.335354</td>
          <td>0.114245</td>
          <td>0.094595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.553494</td>
          <td>0.874566</td>
          <td>26.125811</td>
          <td>0.115376</td>
          <td>25.847361</td>
          <td>0.081099</td>
          <td>25.817223</td>
          <td>0.129066</td>
          <td>25.572595</td>
          <td>0.194563</td>
          <td>25.335325</td>
          <td>0.341547</td>
          <td>0.011587</td>
          <td>0.010297</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.688334</td>
          <td>0.951953</td>
          <td>26.341815</td>
          <td>0.139287</td>
          <td>25.457515</td>
          <td>0.057529</td>
          <td>25.009162</td>
          <td>0.063596</td>
          <td>25.005268</td>
          <td>0.119870</td>
          <td>24.865972</td>
          <td>0.233941</td>
          <td>0.030005</td>
          <td>0.016119</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.827372</td>
          <td>0.556951</td>
          <td>27.779947</td>
          <td>0.473534</td>
          <td>25.858448</td>
          <td>0.087636</td>
          <td>25.215888</td>
          <td>0.081770</td>
          <td>24.672848</td>
          <td>0.095744</td>
          <td>23.884873</td>
          <td>0.108188</td>
          <td>0.183101</td>
          <td>0.100577</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.861041</td>
          <td>0.217813</td>
          <td>26.417884</td>
          <td>0.134760</td>
          <td>26.331856</td>
          <td>0.202128</td>
          <td>25.404261</td>
          <td>0.170258</td>
          <td>27.078537</td>
          <td>1.140461</td>
          <td>0.064761</td>
          <td>0.040176</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.791994</td>
          <td>0.534137</td>
          <td>26.157130</td>
          <td>0.123149</td>
          <td>26.100551</td>
          <td>0.105684</td>
          <td>26.144517</td>
          <td>0.178337</td>
          <td>25.739475</td>
          <td>0.232844</td>
          <td>25.194910</td>
          <td>0.317779</td>
          <td>0.124022</td>
          <td>0.104577</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.428867</td>
          <td>0.406883</td>
          <td>26.639065</td>
          <td>0.185860</td>
          <td>26.813772</td>
          <td>0.194846</td>
          <td>26.490385</td>
          <td>0.237949</td>
          <td>25.579989</td>
          <td>0.203632</td>
          <td>26.176643</td>
          <td>0.661044</td>
          <td>0.124695</td>
          <td>0.099984</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>31.541201</td>
          <td>4.110518</td>
          <td>26.690472</td>
          <td>0.163159</td>
          <td>25.992988</td>
          <td>0.078416</td>
          <td>25.180634</td>
          <td>0.062364</td>
          <td>24.744719</td>
          <td>0.081111</td>
          <td>24.177499</td>
          <td>0.110479</td>
          <td>0.008200</td>
          <td>0.006069</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.961078</td>
          <td>1.770957</td>
          <td>27.429714</td>
          <td>0.310204</td>
          <td>26.759706</td>
          <td>0.158514</td>
          <td>26.372465</td>
          <td>0.182928</td>
          <td>25.973822</td>
          <td>0.241476</td>
          <td>25.806345</td>
          <td>0.438981</td>
          <td>0.056676</td>
          <td>0.049429</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.782869</td>
          <td>1.448218</td>
          <td>27.741018</td>
          <td>0.344805</td>
          <td>25.967201</td>
          <td>0.124699</td>
          <td>25.184307</td>
          <td>0.119393</td>
          <td>24.291099</td>
          <td>0.122137</td>
          <td>0.014065</td>
          <td>0.010573</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.382571</td>
          <td>2.026191</td>
          <td>27.436327</td>
          <td>0.303672</td>
          <td>26.718322</td>
          <td>0.267232</td>
          <td>25.408213</td>
          <td>0.164073</td>
          <td>25.800314</td>
          <td>0.474258</td>
          <td>0.114245</td>
          <td>0.094595</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.068596</td>
          <td>1.102482</td>
          <td>25.990711</td>
          <td>0.089000</td>
          <td>25.892460</td>
          <td>0.071816</td>
          <td>25.571245</td>
          <td>0.088167</td>
          <td>25.520408</td>
          <td>0.159466</td>
          <td>24.857823</td>
          <td>0.198250</td>
          <td>0.011587</td>
          <td>0.010297</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.662633</td>
          <td>0.429240</td>
          <td>26.232180</td>
          <td>0.110535</td>
          <td>25.423687</td>
          <td>0.047681</td>
          <td>25.246625</td>
          <td>0.066621</td>
          <td>24.744476</td>
          <td>0.081674</td>
          <td>24.565519</td>
          <td>0.155657</td>
          <td>0.030005</td>
          <td>0.016119</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.189726</td>
          <td>0.713974</td>
          <td>27.377552</td>
          <td>0.345449</td>
          <td>26.044213</td>
          <td>0.102229</td>
          <td>25.063266</td>
          <td>0.070777</td>
          <td>24.814745</td>
          <td>0.107439</td>
          <td>24.411231</td>
          <td>0.168809</td>
          <td>0.183101</td>
          <td>0.100577</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.885579</td>
          <td>0.516055</td>
          <td>26.605347</td>
          <td>0.156549</td>
          <td>26.370371</td>
          <td>0.113321</td>
          <td>26.486241</td>
          <td>0.201480</td>
          <td>25.839744</td>
          <td>0.216205</td>
          <td>25.751926</td>
          <td>0.421459</td>
          <td>0.064761</td>
          <td>0.040176</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.579200</td>
          <td>0.440889</td>
          <td>26.126280</td>
          <td>0.114352</td>
          <td>26.224333</td>
          <td>0.111708</td>
          <td>25.704604</td>
          <td>0.115792</td>
          <td>25.823612</td>
          <td>0.237508</td>
          <td>25.935775</td>
          <td>0.533971</td>
          <td>0.124022</td>
          <td>0.104577</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.378000</td>
          <td>0.376876</td>
          <td>26.735508</td>
          <td>0.192089</td>
          <td>26.796294</td>
          <td>0.181972</td>
          <td>26.409639</td>
          <td>0.210680</td>
          <td>26.556848</td>
          <td>0.424230</td>
          <td>25.906836</td>
          <td>0.520978</td>
          <td>0.124695</td>
          <td>0.099984</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
