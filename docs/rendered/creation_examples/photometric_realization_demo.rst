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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f7aa8e16bc0>



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
          <td>26.657390</td>
          <td>0.425534</td>
          <td>26.654728</td>
          <td>0.158161</td>
          <td>26.078990</td>
          <td>0.084536</td>
          <td>25.382494</td>
          <td>0.074515</td>
          <td>24.997951</td>
          <td>0.101263</td>
          <td>24.773585</td>
          <td>0.184358</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.663436</td>
          <td>1.516188</td>
          <td>28.060002</td>
          <td>0.490507</td>
          <td>28.013068</td>
          <td>0.425003</td>
          <td>26.829427</td>
          <td>0.258041</td>
          <td>26.553855</td>
          <td>0.372026</td>
          <td>25.876133</td>
          <td>0.447750</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.497307</td>
          <td>0.376239</td>
          <td>26.072575</td>
          <td>0.095494</td>
          <td>24.788109</td>
          <td>0.026996</td>
          <td>23.874133</td>
          <td>0.019792</td>
          <td>23.111264</td>
          <td>0.019309</td>
          <td>22.850499</td>
          <td>0.034171</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.072586</td>
          <td>0.495095</td>
          <td>27.671222</td>
          <td>0.325647</td>
          <td>26.545872</td>
          <td>0.203984</td>
          <td>25.895628</td>
          <td>0.218575</td>
          <td>25.118044</td>
          <td>0.245789</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.452026</td>
          <td>0.363195</td>
          <td>25.735464</td>
          <td>0.070973</td>
          <td>25.426216</td>
          <td>0.047412</td>
          <td>24.739172</td>
          <td>0.042114</td>
          <td>24.210153</td>
          <td>0.050478</td>
          <td>23.623483</td>
          <td>0.067801</td>
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
          <td>25.921373</td>
          <td>0.237011</td>
          <td>26.272291</td>
          <td>0.113698</td>
          <td>26.110718</td>
          <td>0.086931</td>
          <td>25.976860</td>
          <td>0.125475</td>
          <td>26.155594</td>
          <td>0.270792</td>
          <td>25.812371</td>
          <td>0.426629</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.458820</td>
          <td>0.365128</td>
          <td>26.868849</td>
          <td>0.189696</td>
          <td>26.475018</td>
          <td>0.119580</td>
          <td>26.329259</td>
          <td>0.169866</td>
          <td>25.532240</td>
          <td>0.160832</td>
          <td>25.820288</td>
          <td>0.429207</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.460350</td>
          <td>0.755022</td>
          <td>27.076888</td>
          <td>0.225771</td>
          <td>27.207233</td>
          <td>0.223193</td>
          <td>26.061239</td>
          <td>0.134982</td>
          <td>26.207922</td>
          <td>0.282550</td>
          <td>25.262218</td>
          <td>0.276551</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.544944</td>
          <td>0.330256</td>
          <td>26.548019</td>
          <td>0.127402</td>
          <td>25.795177</td>
          <td>0.107118</td>
          <td>25.683138</td>
          <td>0.182851</td>
          <td>25.629207</td>
          <td>0.370463</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.908282</td>
          <td>0.513279</td>
          <td>26.516541</td>
          <td>0.140479</td>
          <td>26.205169</td>
          <td>0.094459</td>
          <td>25.553025</td>
          <td>0.086616</td>
          <td>25.182763</td>
          <td>0.118986</td>
          <td>25.412379</td>
          <td>0.312140</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.668196</td>
          <td>0.183720</td>
          <td>25.906353</td>
          <td>0.085394</td>
          <td>25.343383</td>
          <td>0.085274</td>
          <td>25.138815</td>
          <td>0.134316</td>
          <td>24.834413</td>
          <td>0.227475</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.914835</td>
          <td>0.568187</td>
          <td>30.530635</td>
          <td>2.180912</td>
          <td>27.850812</td>
          <td>0.432076</td>
          <td>26.888562</td>
          <td>0.316121</td>
          <td>26.296529</td>
          <td>0.351216</td>
          <td>25.667167</td>
          <td>0.441415</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.789470</td>
          <td>0.526234</td>
          <td>26.077639</td>
          <td>0.112837</td>
          <td>24.819890</td>
          <td>0.033389</td>
          <td>23.846721</td>
          <td>0.023330</td>
          <td>23.165406</td>
          <td>0.024217</td>
          <td>22.886609</td>
          <td>0.042750</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.632879</td>
          <td>0.855026</td>
          <td>27.616579</td>
          <td>0.382693</td>
          <td>27.453000</td>
          <td>0.517661</td>
          <td>26.177464</td>
          <td>0.339515</td>
          <td>25.330125</td>
          <td>0.361642</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.842469</td>
          <td>0.248004</td>
          <td>25.888849</td>
          <td>0.093805</td>
          <td>25.390377</td>
          <td>0.054114</td>
          <td>24.776149</td>
          <td>0.051635</td>
          <td>24.360744</td>
          <td>0.067948</td>
          <td>23.722199</td>
          <td>0.087542</td>
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
          <td>26.113355</td>
          <td>0.313365</td>
          <td>26.250573</td>
          <td>0.130942</td>
          <td>26.111658</td>
          <td>0.104423</td>
          <td>26.397310</td>
          <td>0.215882</td>
          <td>25.964390</td>
          <td>0.274400</td>
          <td>25.548483</td>
          <td>0.410804</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.106460</td>
          <td>0.651845</td>
          <td>27.486664</td>
          <td>0.359770</td>
          <td>26.593837</td>
          <td>0.155919</td>
          <td>26.319939</td>
          <td>0.199015</td>
          <td>25.695577</td>
          <td>0.216460</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.290065</td>
          <td>0.310072</td>
          <td>27.080607</td>
          <td>0.236867</td>
          <td>26.635238</td>
          <td>0.260686</td>
          <td>26.218363</td>
          <td>0.333950</td>
          <td>24.815091</td>
          <td>0.226651</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.393412</td>
          <td>0.803025</td>
          <td>28.208451</td>
          <td>0.628078</td>
          <td>26.410693</td>
          <td>0.136810</td>
          <td>25.926872</td>
          <td>0.146415</td>
          <td>25.915246</td>
          <td>0.266189</td>
          <td>25.843459</td>
          <td>0.517176</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.048690</td>
          <td>0.295274</td>
          <td>26.742812</td>
          <td>0.197350</td>
          <td>26.148134</td>
          <td>0.106635</td>
          <td>25.610347</td>
          <td>0.108893</td>
          <td>25.518101</td>
          <td>0.187555</td>
          <td>24.631017</td>
          <td>0.193796</td>
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
          <td>26.213946</td>
          <td>0.300784</td>
          <td>26.943398</td>
          <td>0.201992</td>
          <td>26.027181</td>
          <td>0.080772</td>
          <td>25.340142</td>
          <td>0.071786</td>
          <td>24.880495</td>
          <td>0.091360</td>
          <td>24.716424</td>
          <td>0.175665</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.093208</td>
          <td>0.503034</td>
          <td>27.666228</td>
          <td>0.324634</td>
          <td>28.456352</td>
          <td>0.856524</td>
          <td>26.640768</td>
          <td>0.398279</td>
          <td>25.942985</td>
          <td>0.471189</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.214363</td>
          <td>0.668876</td>
          <td>26.029286</td>
          <td>0.098791</td>
          <td>24.789150</td>
          <td>0.029329</td>
          <td>23.856292</td>
          <td>0.021192</td>
          <td>23.119193</td>
          <td>0.021051</td>
          <td>22.844224</td>
          <td>0.037020</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.325088</td>
          <td>0.643155</td>
          <td>27.677031</td>
          <td>0.606311</td>
          <td>26.451250</td>
          <td>0.418709</td>
          <td>25.041185</td>
          <td>0.286382</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.234322</td>
          <td>0.305988</td>
          <td>25.646483</td>
          <td>0.065686</td>
          <td>25.408591</td>
          <td>0.046743</td>
          <td>24.759856</td>
          <td>0.042958</td>
          <td>24.397598</td>
          <td>0.059701</td>
          <td>23.620445</td>
          <td>0.067720</td>
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
          <td>26.373756</td>
          <td>0.359047</td>
          <td>26.285906</td>
          <td>0.123182</td>
          <td>26.230748</td>
          <td>0.104496</td>
          <td>26.065338</td>
          <td>0.146856</td>
          <td>25.685935</td>
          <td>0.197652</td>
          <td>25.483515</td>
          <td>0.355439</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.171073</td>
          <td>0.625455</td>
          <td>26.843259</td>
          <td>0.188216</td>
          <td>26.692120</td>
          <td>0.146630</td>
          <td>26.232455</td>
          <td>0.159050</td>
          <td>25.948813</td>
          <td>0.232030</td>
          <td>25.990478</td>
          <td>0.494814</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.241859</td>
          <td>0.579976</td>
          <td>26.814454</td>
          <td>0.168037</td>
          <td>26.226762</td>
          <td>0.163590</td>
          <td>25.745552</td>
          <td>0.201972</td>
          <td>25.874748</td>
          <td>0.467101</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.981036</td>
          <td>0.229350</td>
          <td>26.639838</td>
          <td>0.154308</td>
          <td>26.134267</td>
          <td>0.161579</td>
          <td>25.501443</td>
          <td>0.175065</td>
          <td>25.310941</td>
          <td>0.320614</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.715146</td>
          <td>0.172088</td>
          <td>26.141459</td>
          <td>0.092865</td>
          <td>25.620709</td>
          <td>0.095759</td>
          <td>25.036799</td>
          <td>0.108897</td>
          <td>24.830489</td>
          <td>0.201050</td>
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
