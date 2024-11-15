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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7effd97a45e0>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>27.362144</td>
          <td>0.706951</td>
          <td>26.464106</td>
          <td>0.134270</td>
          <td>26.171958</td>
          <td>0.091743</td>
          <td>25.407602</td>
          <td>0.076187</td>
          <td>24.817621</td>
          <td>0.086432</td>
          <td>24.942419</td>
          <td>0.212470</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.849827</td>
          <td>0.491650</td>
          <td>28.186051</td>
          <td>0.538008</td>
          <td>28.311835</td>
          <td>0.531004</td>
          <td>27.155793</td>
          <td>0.335660</td>
          <td>27.641529</td>
          <td>0.811979</td>
          <td>26.061582</td>
          <td>0.513996</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.835456</td>
          <td>0.958989</td>
          <td>25.932442</td>
          <td>0.084436</td>
          <td>24.794005</td>
          <td>0.027135</td>
          <td>23.826020</td>
          <td>0.019005</td>
          <td>23.154486</td>
          <td>0.020027</td>
          <td>22.848774</td>
          <td>0.034119</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.730786</td>
          <td>0.898852</td>
          <td>28.709635</td>
          <td>0.773329</td>
          <td>27.310929</td>
          <td>0.243201</td>
          <td>26.542808</td>
          <td>0.203460</td>
          <td>26.482652</td>
          <td>0.351858</td>
          <td>24.986508</td>
          <td>0.220427</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.440531</td>
          <td>0.359945</td>
          <td>25.756434</td>
          <td>0.072300</td>
          <td>25.447755</td>
          <td>0.048327</td>
          <td>24.781477</td>
          <td>0.043724</td>
          <td>24.352040</td>
          <td>0.057254</td>
          <td>23.708859</td>
          <td>0.073122</td>
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
          <td>2.147172</td>
          <td>26.753329</td>
          <td>0.457535</td>
          <td>26.426266</td>
          <td>0.129950</td>
          <td>26.151231</td>
          <td>0.090086</td>
          <td>26.166488</td>
          <td>0.147793</td>
          <td>26.229489</td>
          <td>0.287526</td>
          <td>26.498440</td>
          <td>0.699884</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.360342</td>
          <td>2.077625</td>
          <td>27.557625</td>
          <td>0.333592</td>
          <td>26.618702</td>
          <td>0.135436</td>
          <td>26.318015</td>
          <td>0.168248</td>
          <td>26.193250</td>
          <td>0.279209</td>
          <td>25.403701</td>
          <td>0.309979</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.344811</td>
          <td>0.698697</td>
          <td>27.373832</td>
          <td>0.287966</td>
          <td>26.642566</td>
          <td>0.138254</td>
          <td>26.230452</td>
          <td>0.156127</td>
          <td>25.926230</td>
          <td>0.224213</td>
          <td>25.382853</td>
          <td>0.304843</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.808044</td>
          <td>1.626741</td>
          <td>27.221157</td>
          <td>0.254311</td>
          <td>26.461330</td>
          <td>0.118165</td>
          <td>25.969516</td>
          <td>0.124678</td>
          <td>25.552841</td>
          <td>0.163685</td>
          <td>25.339094</td>
          <td>0.294301</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.437949</td>
          <td>0.743861</td>
          <td>26.738191</td>
          <td>0.169825</td>
          <td>26.120301</td>
          <td>0.087668</td>
          <td>25.465466</td>
          <td>0.080181</td>
          <td>25.330700</td>
          <td>0.135262</td>
          <td>24.466400</td>
          <td>0.141827</td>
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
          <td>0.890625</td>
          <td>27.240050</td>
          <td>0.712398</td>
          <td>26.930036</td>
          <td>0.228754</td>
          <td>26.180174</td>
          <td>0.108572</td>
          <td>25.342140</td>
          <td>0.085181</td>
          <td>25.189844</td>
          <td>0.140361</td>
          <td>25.148929</td>
          <td>0.294242</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.818758</td>
          <td>0.421662</td>
          <td>26.914393</td>
          <td>0.322699</td>
          <td>26.794115</td>
          <td>0.512885</td>
          <td>25.803979</td>
          <td>0.489037</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.361009</td>
          <td>0.782141</td>
          <td>25.984329</td>
          <td>0.104021</td>
          <td>24.788929</td>
          <td>0.032492</td>
          <td>23.910571</td>
          <td>0.024652</td>
          <td>23.153868</td>
          <td>0.023977</td>
          <td>22.844847</td>
          <td>0.041198</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.179771</td>
          <td>0.711747</td>
          <td>28.080437</td>
          <td>0.589044</td>
          <td>27.871450</td>
          <td>0.464787</td>
          <td>26.451995</td>
          <td>0.236311</td>
          <td>25.980214</td>
          <td>0.290009</td>
          <td>24.995349</td>
          <td>0.276872</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.961619</td>
          <td>0.273342</td>
          <td>25.700610</td>
          <td>0.079504</td>
          <td>25.446641</td>
          <td>0.056884</td>
          <td>24.807087</td>
          <td>0.053072</td>
          <td>24.393786</td>
          <td>0.069963</td>
          <td>23.655773</td>
          <td>0.082569</td>
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
          <td>2.147172</td>
          <td>27.391416</td>
          <td>0.797239</td>
          <td>26.215737</td>
          <td>0.127056</td>
          <td>26.103424</td>
          <td>0.103674</td>
          <td>26.108678</td>
          <td>0.169264</td>
          <td>26.444009</td>
          <td>0.401228</td>
          <td>26.222500</td>
          <td>0.671065</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.779769</td>
          <td>1.007409</td>
          <td>26.706785</td>
          <td>0.190489</td>
          <td>26.841781</td>
          <td>0.192474</td>
          <td>26.726430</td>
          <td>0.278482</td>
          <td>25.618592</td>
          <td>0.202964</td>
          <td>25.874217</td>
          <td>0.516776</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.789565</td>
          <td>1.017876</td>
          <td>27.150183</td>
          <td>0.277011</td>
          <td>26.684914</td>
          <td>0.169938</td>
          <td>26.378167</td>
          <td>0.210751</td>
          <td>25.903534</td>
          <td>0.259130</td>
          <td>25.867905</td>
          <td>0.518198</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.103106</td>
          <td>0.661082</td>
          <td>27.146444</td>
          <td>0.280505</td>
          <td>26.657506</td>
          <td>0.169044</td>
          <td>25.754800</td>
          <td>0.126207</td>
          <td>25.594764</td>
          <td>0.204185</td>
          <td>25.410675</td>
          <td>0.372794</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.296123</td>
          <td>0.359347</td>
          <td>26.486692</td>
          <td>0.158847</td>
          <td>26.150026</td>
          <td>0.106812</td>
          <td>25.500747</td>
          <td>0.098940</td>
          <td>25.398432</td>
          <td>0.169463</td>
          <td>25.065160</td>
          <td>0.277586</td>
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.478082</td>
          <td>0.135914</td>
          <td>26.043676</td>
          <td>0.081955</td>
          <td>25.249610</td>
          <td>0.066256</td>
          <td>25.056334</td>
          <td>0.106583</td>
          <td>25.066003</td>
          <td>0.235488</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.306949</td>
          <td>0.587228</td>
          <td>27.715865</td>
          <td>0.337672</td>
          <td>26.879494</td>
          <td>0.269064</td>
          <td>26.177507</td>
          <td>0.275903</td>
          <td>27.129145</td>
          <td>1.045661</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.767230</td>
          <td>0.485859</td>
          <td>25.869643</td>
          <td>0.085880</td>
          <td>24.781046</td>
          <td>0.029122</td>
          <td>23.850442</td>
          <td>0.021086</td>
          <td>23.127015</td>
          <td>0.021191</td>
          <td>22.792625</td>
          <td>0.035370</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.274485</td>
          <td>1.369365</td>
          <td>28.127155</td>
          <td>0.607252</td>
          <td>27.348504</td>
          <td>0.308779</td>
          <td>26.371028</td>
          <td>0.220193</td>
          <td>26.117459</td>
          <td>0.322693</td>
          <td>25.330008</td>
          <td>0.360433</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.495454</td>
          <td>0.376025</td>
          <td>25.755008</td>
          <td>0.072298</td>
          <td>25.430100</td>
          <td>0.047644</td>
          <td>24.883426</td>
          <td>0.047938</td>
          <td>24.322717</td>
          <td>0.055863</td>
          <td>23.671503</td>
          <td>0.070851</td>
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
          <td>2.147172</td>
          <td>26.563435</td>
          <td>0.415788</td>
          <td>26.375707</td>
          <td>0.133135</td>
          <td>26.051437</td>
          <td>0.089289</td>
          <td>26.226271</td>
          <td>0.168532</td>
          <td>25.843105</td>
          <td>0.225397</td>
          <td>25.686720</td>
          <td>0.416060</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.592543</td>
          <td>0.830187</td>
          <td>27.086850</td>
          <td>0.230741</td>
          <td>26.614252</td>
          <td>0.137119</td>
          <td>26.323563</td>
          <td>0.171898</td>
          <td>25.955733</td>
          <td>0.233363</td>
          <td>25.546819</td>
          <td>0.352654</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.792980</td>
          <td>0.957050</td>
          <td>27.354159</td>
          <td>0.294766</td>
          <td>26.803561</td>
          <td>0.166485</td>
          <td>26.248281</td>
          <td>0.166619</td>
          <td>25.670657</td>
          <td>0.189636</td>
          <td>26.018433</td>
          <td>0.519504</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.371489</td>
          <td>0.315184</td>
          <td>26.525169</td>
          <td>0.139829</td>
          <td>25.968848</td>
          <td>0.140200</td>
          <td>25.483785</td>
          <td>0.172459</td>
          <td>24.860173</td>
          <td>0.222029</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.365937</td>
          <td>0.723734</td>
          <td>26.559643</td>
          <td>0.150698</td>
          <td>26.133038</td>
          <td>0.092180</td>
          <td>25.601344</td>
          <td>0.094145</td>
          <td>25.301131</td>
          <td>0.136987</td>
          <td>24.714376</td>
          <td>0.182303</td>
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
