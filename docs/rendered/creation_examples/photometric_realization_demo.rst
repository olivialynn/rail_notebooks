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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f9243bf0ca0>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>26.616511</td>
          <td>0.412466</td>
          <td>26.984836</td>
          <td>0.209102</td>
          <td>26.053489</td>
          <td>0.082657</td>
          <td>25.226299</td>
          <td>0.064892</td>
          <td>24.634407</td>
          <td>0.073529</td>
          <td>23.959283</td>
          <td>0.091191</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.068752</td>
          <td>0.576477</td>
          <td>27.883064</td>
          <td>0.429511</td>
          <td>26.700300</td>
          <td>0.145303</td>
          <td>26.256028</td>
          <td>0.159581</td>
          <td>25.638370</td>
          <td>0.176044</td>
          <td>25.103757</td>
          <td>0.242913</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.727163</td>
          <td>0.896815</td>
          <td>27.854784</td>
          <td>0.420359</td>
          <td>27.573010</td>
          <td>0.301056</td>
          <td>26.079408</td>
          <td>0.137116</td>
          <td>25.066726</td>
          <td>0.107541</td>
          <td>24.248066</td>
          <td>0.117399</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.487128</td>
          <td>0.768515</td>
          <td>27.758920</td>
          <td>0.390520</td>
          <td>27.408898</td>
          <td>0.263562</td>
          <td>26.292974</td>
          <td>0.164695</td>
          <td>25.341500</td>
          <td>0.136529</td>
          <td>25.123188</td>
          <td>0.246832</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.538592</td>
          <td>0.388473</td>
          <td>26.115125</td>
          <td>0.099120</td>
          <td>25.938842</td>
          <td>0.074700</td>
          <td>25.676410</td>
          <td>0.096538</td>
          <td>25.259733</td>
          <td>0.127207</td>
          <td>25.295770</td>
          <td>0.284179</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>29.184764</td>
          <td>1.929826</td>
          <td>26.310131</td>
          <td>0.117502</td>
          <td>25.432669</td>
          <td>0.047684</td>
          <td>25.007288</td>
          <td>0.053431</td>
          <td>24.783928</td>
          <td>0.083905</td>
          <td>24.431531</td>
          <td>0.137628</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.688999</td>
          <td>0.162857</td>
          <td>26.083243</td>
          <td>0.084853</td>
          <td>25.052007</td>
          <td>0.055595</td>
          <td>25.025609</td>
          <td>0.103744</td>
          <td>23.976058</td>
          <td>0.092546</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.934465</td>
          <td>0.523205</td>
          <td>26.719790</td>
          <td>0.167187</td>
          <td>26.659293</td>
          <td>0.140263</td>
          <td>26.550217</td>
          <td>0.204728</td>
          <td>26.352585</td>
          <td>0.317410</td>
          <td>26.171851</td>
          <td>0.556891</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.881917</td>
          <td>0.503433</td>
          <td>26.224984</td>
          <td>0.109107</td>
          <td>26.021729</td>
          <td>0.080373</td>
          <td>26.228640</td>
          <td>0.155885</td>
          <td>25.858335</td>
          <td>0.211879</td>
          <td>25.172078</td>
          <td>0.256943</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.548226</td>
          <td>0.799918</td>
          <td>27.232014</td>
          <td>0.256584</td>
          <td>26.683244</td>
          <td>0.143186</td>
          <td>26.173715</td>
          <td>0.148713</td>
          <td>26.314038</td>
          <td>0.307775</td>
          <td>25.149990</td>
          <td>0.252330</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>26.825685</td>
          <td>0.532713</td>
          <td>26.699138</td>
          <td>0.188582</td>
          <td>26.065315</td>
          <td>0.098193</td>
          <td>25.113078</td>
          <td>0.069585</td>
          <td>24.742671</td>
          <td>0.095120</td>
          <td>23.924177</td>
          <td>0.104475</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.356815</td>
          <td>1.388625</td>
          <td>27.036863</td>
          <td>0.249877</td>
          <td>26.501514</td>
          <td>0.143485</td>
          <td>26.308210</td>
          <td>0.196290</td>
          <td>25.721106</td>
          <td>0.220284</td>
          <td>25.747372</td>
          <td>0.468858</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.255651</td>
          <td>1.206031</td>
          <td>28.427680</td>
          <td>0.668382</td>
          <td>26.004532</td>
          <td>0.155067</td>
          <td>25.124178</td>
          <td>0.135568</td>
          <td>24.308490</td>
          <td>0.149110</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.902965</td>
          <td>0.518270</td>
          <td>27.341766</td>
          <td>0.308118</td>
          <td>26.018813</td>
          <td>0.164202</td>
          <td>25.604812</td>
          <td>0.213028</td>
          <td>26.582010</td>
          <td>0.881173</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.697219</td>
          <td>0.484845</td>
          <td>26.040187</td>
          <td>0.107079</td>
          <td>26.048670</td>
          <td>0.096802</td>
          <td>25.578144</td>
          <td>0.104819</td>
          <td>25.388692</td>
          <td>0.166490</td>
          <td>25.042896</td>
          <td>0.270099</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>28.469642</td>
          <td>1.484598</td>
          <td>26.252270</td>
          <td>0.131134</td>
          <td>25.388364</td>
          <td>0.055161</td>
          <td>25.028475</td>
          <td>0.065996</td>
          <td>24.833202</td>
          <td>0.105146</td>
          <td>25.296110</td>
          <td>0.337490</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.535541</td>
          <td>0.164757</td>
          <td>25.951081</td>
          <td>0.089195</td>
          <td>25.104233</td>
          <td>0.069344</td>
          <td>24.790747</td>
          <td>0.099628</td>
          <td>24.193658</td>
          <td>0.132623</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.107592</td>
          <td>1.222101</td>
          <td>26.784936</td>
          <td>0.204934</td>
          <td>26.323885</td>
          <td>0.124604</td>
          <td>26.498067</td>
          <td>0.232858</td>
          <td>25.965909</td>
          <td>0.272661</td>
          <td>24.928799</td>
          <td>0.248973</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.884059</td>
          <td>0.566719</td>
          <td>25.917005</td>
          <td>0.098886</td>
          <td>25.972790</td>
          <td>0.093427</td>
          <td>25.745261</td>
          <td>0.125167</td>
          <td>25.926781</td>
          <td>0.268705</td>
          <td>24.491619</td>
          <td>0.175931</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.286840</td>
          <td>0.356745</td>
          <td>26.975625</td>
          <td>0.239576</td>
          <td>26.597162</td>
          <td>0.157260</td>
          <td>26.278959</td>
          <td>0.193392</td>
          <td>25.734239</td>
          <td>0.224783</td>
          <td>25.339654</td>
          <td>0.345782</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>28.163737</td>
          <td>1.163315</td>
          <td>26.941979</td>
          <td>0.201752</td>
          <td>26.038751</td>
          <td>0.081600</td>
          <td>25.103845</td>
          <td>0.058221</td>
          <td>24.728550</td>
          <td>0.079916</td>
          <td>24.083437</td>
          <td>0.101699</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.258908</td>
          <td>0.262492</td>
          <td>26.742238</td>
          <td>0.150773</td>
          <td>26.305630</td>
          <td>0.166643</td>
          <td>25.954352</td>
          <td>0.229713</td>
          <td>24.953549</td>
          <td>0.214654</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.038764</td>
          <td>1.125203</td>
          <td>30.700144</td>
          <td>2.258047</td>
          <td>28.920162</td>
          <td>0.857688</td>
          <td>25.985724</td>
          <td>0.137552</td>
          <td>25.235312</td>
          <td>0.134993</td>
          <td>24.546513</td>
          <td>0.165000</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.855777</td>
          <td>0.499244</td>
          <td>27.344456</td>
          <td>0.307779</td>
          <td>26.278010</td>
          <td>0.203730</td>
          <td>26.180538</td>
          <td>0.339250</td>
          <td>24.895984</td>
          <td>0.254440</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.324976</td>
          <td>0.689861</td>
          <td>25.990399</td>
          <td>0.088959</td>
          <td>25.960403</td>
          <td>0.076246</td>
          <td>25.861608</td>
          <td>0.113679</td>
          <td>25.350015</td>
          <td>0.137728</td>
          <td>25.763619</td>
          <td>0.411564</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>28.602555</td>
          <td>1.518153</td>
          <td>26.454113</td>
          <td>0.142443</td>
          <td>25.400872</td>
          <td>0.050207</td>
          <td>25.146399</td>
          <td>0.065703</td>
          <td>24.921107</td>
          <td>0.102398</td>
          <td>24.798451</td>
          <td>0.203567</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.747424</td>
          <td>0.459998</td>
          <td>26.784881</td>
          <td>0.179153</td>
          <td>25.985228</td>
          <td>0.079126</td>
          <td>25.163379</td>
          <td>0.062453</td>
          <td>24.794591</td>
          <td>0.086107</td>
          <td>24.229426</td>
          <td>0.117485</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.726336</td>
          <td>0.461741</td>
          <td>26.510063</td>
          <td>0.145669</td>
          <td>26.447815</td>
          <td>0.122577</td>
          <td>26.629014</td>
          <td>0.229520</td>
          <td>26.136189</td>
          <td>0.278874</td>
          <td>25.744167</td>
          <td>0.423230</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.194897</td>
          <td>0.319227</td>
          <td>26.219086</td>
          <td>0.119971</td>
          <td>26.091880</td>
          <td>0.095907</td>
          <td>25.851918</td>
          <td>0.126724</td>
          <td>25.924959</td>
          <td>0.249451</td>
          <td>25.813425</td>
          <td>0.472697</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.100566</td>
          <td>0.602777</td>
          <td>26.942975</td>
          <td>0.208532</td>
          <td>26.624614</td>
          <td>0.141430</td>
          <td>26.679794</td>
          <td>0.237030</td>
          <td>26.494543</td>
          <td>0.367668</td>
          <td>25.830033</td>
          <td>0.447693</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
