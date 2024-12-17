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

    <pzflow.flow.Flow at 0x7ffaa701ccd0>



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
          <td>28.285028</td>
          <td>1.244473</td>
          <td>26.681455</td>
          <td>0.161812</td>
          <td>26.072954</td>
          <td>0.084087</td>
          <td>25.284767</td>
          <td>0.068342</td>
          <td>25.006538</td>
          <td>0.102027</td>
          <td>25.130882</td>
          <td>0.248400</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.101505</td>
          <td>0.505768</td>
          <td>27.585941</td>
          <td>0.304199</td>
          <td>27.512388</td>
          <td>0.442419</td>
          <td>27.254012</td>
          <td>0.625202</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.483339</td>
          <td>0.372174</td>
          <td>25.971884</td>
          <td>0.087416</td>
          <td>24.763883</td>
          <td>0.026432</td>
          <td>23.891476</td>
          <td>0.020085</td>
          <td>23.129793</td>
          <td>0.019613</td>
          <td>22.868645</td>
          <td>0.034723</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.615933</td>
          <td>0.412283</td>
          <td>27.807192</td>
          <td>0.405319</td>
          <td>27.590570</td>
          <td>0.305331</td>
          <td>26.907596</td>
          <td>0.275036</td>
          <td>26.377801</td>
          <td>0.323853</td>
          <td>24.994076</td>
          <td>0.221820</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.428747</td>
          <td>0.356639</td>
          <td>25.772354</td>
          <td>0.073323</td>
          <td>25.381042</td>
          <td>0.045548</td>
          <td>24.778297</td>
          <td>0.043601</td>
          <td>24.435778</td>
          <td>0.061669</td>
          <td>23.613359</td>
          <td>0.067196</td>
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
          <td>25.968868</td>
          <td>0.246463</td>
          <td>26.477380</td>
          <td>0.135817</td>
          <td>26.286291</td>
          <td>0.101423</td>
          <td>26.203887</td>
          <td>0.152614</td>
          <td>26.143881</td>
          <td>0.268220</td>
          <td>25.294911</td>
          <td>0.283982</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.024814</td>
          <td>1.073970</td>
          <td>26.608534</td>
          <td>0.152030</td>
          <td>26.944375</td>
          <td>0.178981</td>
          <td>26.401174</td>
          <td>0.180562</td>
          <td>26.022248</td>
          <td>0.242761</td>
          <td>25.373159</td>
          <td>0.302480</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.998831</td>
          <td>0.548243</td>
          <td>28.076158</td>
          <td>0.496404</td>
          <td>26.821272</td>
          <td>0.161179</td>
          <td>26.340968</td>
          <td>0.171567</td>
          <td>25.784219</td>
          <td>0.199120</td>
          <td>25.280657</td>
          <td>0.280720</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.908255</td>
          <td>1.002260</td>
          <td>27.913384</td>
          <td>0.439504</td>
          <td>26.876043</td>
          <td>0.168886</td>
          <td>25.824359</td>
          <td>0.109883</td>
          <td>25.705153</td>
          <td>0.186287</td>
          <td>25.196198</td>
          <td>0.262065</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.224957</td>
          <td>0.643507</td>
          <td>26.487679</td>
          <td>0.137028</td>
          <td>26.085064</td>
          <td>0.084989</td>
          <td>25.505645</td>
          <td>0.083074</td>
          <td>25.283717</td>
          <td>0.129877</td>
          <td>24.630019</td>
          <td>0.163188</td>
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
          <td>28.283102</td>
          <td>1.335826</td>
          <td>26.929481</td>
          <td>0.228649</td>
          <td>25.928452</td>
          <td>0.087072</td>
          <td>25.284631</td>
          <td>0.080972</td>
          <td>24.806263</td>
          <td>0.100572</td>
          <td>24.800218</td>
          <td>0.221102</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.260430</td>
          <td>1.319957</td>
          <td>27.677165</td>
          <td>0.415618</td>
          <td>27.320974</td>
          <td>0.284981</td>
          <td>27.324411</td>
          <td>0.443689</td>
          <td>27.409170</td>
          <td>0.786648</td>
          <td>25.850378</td>
          <td>0.506084</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.526044</td>
          <td>0.432576</td>
          <td>25.898312</td>
          <td>0.096484</td>
          <td>24.781492</td>
          <td>0.032280</td>
          <td>23.845062</td>
          <td>0.023297</td>
          <td>23.118367</td>
          <td>0.023255</td>
          <td>22.881968</td>
          <td>0.042574</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.286186</td>
          <td>0.764175</td>
          <td>27.555219</td>
          <td>0.399097</td>
          <td>27.122125</td>
          <td>0.257890</td>
          <td>26.895696</td>
          <td>0.338420</td>
          <td>26.218916</td>
          <td>0.350793</td>
          <td>25.170750</td>
          <td>0.318851</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.720272</td>
          <td>0.493194</td>
          <td>25.847232</td>
          <td>0.090443</td>
          <td>25.424422</td>
          <td>0.055774</td>
          <td>24.792356</td>
          <td>0.052383</td>
          <td>24.247243</td>
          <td>0.061449</td>
          <td>23.646299</td>
          <td>0.081882</td>
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
          <td>26.323562</td>
          <td>0.369876</td>
          <td>26.214001</td>
          <td>0.126865</td>
          <td>26.152613</td>
          <td>0.108227</td>
          <td>26.528033</td>
          <td>0.240611</td>
          <td>25.659434</td>
          <td>0.213410</td>
          <td>26.868199</td>
          <td>1.016401</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.567859</td>
          <td>0.441180</td>
          <td>27.355812</td>
          <td>0.324469</td>
          <td>26.817704</td>
          <td>0.188605</td>
          <td>26.517869</td>
          <td>0.234727</td>
          <td>25.936285</td>
          <td>0.264040</td>
          <td>25.845736</td>
          <td>0.506082</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.002573</td>
          <td>0.534959</td>
          <td>27.130290</td>
          <td>0.246773</td>
          <td>26.655454</td>
          <td>0.265028</td>
          <td>26.559251</td>
          <td>0.435045</td>
          <td>25.445416</td>
          <td>0.376581</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.348998</td>
          <td>0.780053</td>
          <td>27.292792</td>
          <td>0.315551</td>
          <td>26.878357</td>
          <td>0.203723</td>
          <td>25.980924</td>
          <td>0.153366</td>
          <td>25.797357</td>
          <td>0.241652</td>
          <td>25.029233</td>
          <td>0.275126</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.029404</td>
          <td>0.619976</td>
          <td>26.503328</td>
          <td>0.161119</td>
          <td>26.132342</td>
          <td>0.105174</td>
          <td>25.645881</td>
          <td>0.112321</td>
          <td>25.504374</td>
          <td>0.185393</td>
          <td>25.058538</td>
          <td>0.276097</td>
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
          <td>26.338386</td>
          <td>0.332160</td>
          <td>26.598211</td>
          <td>0.150708</td>
          <td>26.121921</td>
          <td>0.087805</td>
          <td>25.396572</td>
          <td>0.075459</td>
          <td>24.910552</td>
          <td>0.093804</td>
          <td>24.625182</td>
          <td>0.162537</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.827651</td>
          <td>1.642573</td>
          <td>29.636315</td>
          <td>1.341441</td>
          <td>27.715502</td>
          <td>0.337576</td>
          <td>27.248212</td>
          <td>0.361318</td>
          <td>26.334318</td>
          <td>0.313081</td>
          <td>26.076807</td>
          <td>0.520186</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.805758</td>
          <td>0.499899</td>
          <td>26.154835</td>
          <td>0.110241</td>
          <td>24.798849</td>
          <td>0.029579</td>
          <td>23.867913</td>
          <td>0.021403</td>
          <td>23.128842</td>
          <td>0.021224</td>
          <td>22.827579</td>
          <td>0.036479</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.980198</td>
          <td>1.166923</td>
          <td>27.669147</td>
          <td>0.434173</td>
          <td>27.389722</td>
          <td>0.319119</td>
          <td>26.289585</td>
          <td>0.205716</td>
          <td>27.575983</td>
          <td>0.916240</td>
          <td>24.964845</td>
          <td>0.269173</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.659274</td>
          <td>0.859802</td>
          <td>25.665209</td>
          <td>0.066783</td>
          <td>25.383264</td>
          <td>0.045704</td>
          <td>24.855068</td>
          <td>0.046746</td>
          <td>24.256936</td>
          <td>0.052694</td>
          <td>23.580534</td>
          <td>0.065368</td>
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
          <td>27.033507</td>
          <td>0.588210</td>
          <td>26.425865</td>
          <td>0.139021</td>
          <td>26.242496</td>
          <td>0.105575</td>
          <td>26.013325</td>
          <td>0.140427</td>
          <td>26.221120</td>
          <td>0.306918</td>
          <td>25.555631</td>
          <td>0.376045</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.812231</td>
          <td>2.485233</td>
          <td>26.948596</td>
          <td>0.205643</td>
          <td>26.951372</td>
          <td>0.182926</td>
          <td>26.342940</td>
          <td>0.174752</td>
          <td>26.415014</td>
          <td>0.338544</td>
          <td>24.857751</td>
          <td>0.201180</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.377858</td>
          <td>0.300441</td>
          <td>26.989350</td>
          <td>0.194862</td>
          <td>26.194345</td>
          <td>0.159122</td>
          <td>25.618427</td>
          <td>0.181447</td>
          <td>26.492439</td>
          <td>0.724745</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.339674</td>
          <td>2.139926</td>
          <td>28.248137</td>
          <td>0.610516</td>
          <td>26.540427</td>
          <td>0.141679</td>
          <td>25.986583</td>
          <td>0.142358</td>
          <td>25.749818</td>
          <td>0.215772</td>
          <td>25.250102</td>
          <td>0.305392</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.041556</td>
          <td>0.578052</td>
          <td>26.340802</td>
          <td>0.124788</td>
          <td>26.285295</td>
          <td>0.105341</td>
          <td>25.706787</td>
          <td>0.103261</td>
          <td>25.187677</td>
          <td>0.124178</td>
          <td>24.858818</td>
          <td>0.205883</td>
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
